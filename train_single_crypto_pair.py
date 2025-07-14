"""
Скрипт для обучения модели на одной паре криптовалют из attached_assets
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PricePredictionLSTM(nn.Module):
    """Модель LSTM для прогнозирования цен криптовалют"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Инициализация модели LSTM
        """
        super(PricePredictionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM слои
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Полносвязный слой для прогнозирования
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Прямой проход через модель
        """
        # Инициализация скрытого состояния
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Прямой проход через LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Взять выход последнего временного шага
        out = self.fc(out[:, -1, :])
        
        return out

def load_csv(filepath):
    """
    Загрузка CSV файла с данными криптовалюты
    """
    try:
        # Получение имени символа из имени файла
        filename = os.path.basename(filepath)
        symbol = os.path.splitext(filename)[0]
        
        # Загрузка данных CSV
        df = pd.read_csv(filepath, header=None, names=['timestamp', 'price', 'volume'])
        
        # Конвертация timestamp в datetime
        if df['timestamp'].dtype == np.int64 or df['timestamp'].dtype == np.float64:
            if df['timestamp'].iloc[0] > 1e12:  # миллисекунды
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:  # секунды
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # Сортировка по timestamp
        df = df.sort_values('datetime')
        
        # Установка timestamp как индекса
        df = df.set_index('datetime')
        
        logger.info(f"Загружено {len(df)} строк для {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"Ошибка загрузки {filepath}: {str(e)}")
        return None

def preprocess_data(df):
    """
    Предобработка данных для обучения модели
    """
    try:
        # Удаление дубликатов
        df = df.drop_duplicates()
        
        # Обработка пропущенных значений
        df = df.dropna()
        
        # Убедиться, что цена и объем числовые
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Удаление строк с недопустимыми значениями
        df = df.dropna()
        
        # Расчет признаков
        # Доходности
        df['return'] = df['price'].pct_change(fill_method=None)
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        # Изменения цены
        df['price_change_1'] = df['price'].pct_change(1, fill_method=None)
        df['price_change_6'] = df['price'].pct_change(6, fill_method=None)
        df['price_change_12'] = df['price'].pct_change(12, fill_method=None)
        df['price_change_24'] = df['price'].pct_change(24, fill_method=None)
        
        # Скользящие средние
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_10'] = df['price'].rolling(window=10).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        
        # Экспоненциальные скользящие средние
        df['ema_5'] = df['price'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['price'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['price'].ewm(span=20, adjust=False).mean()
        
        # Признаки объема
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        
        # Волатильность цены
        df['volatility_5'] = df['return'].rolling(window=5).std()
        df['volatility_10'] = df['return'].rolling(window=10).std()
        
        # Нормализованная цена и объем
        min_price = df['price'].min()
        max_price = df['price'].max()
        if max_price > min_price:
            df['price_normalized'] = (df['price'] - min_price) / (max_price - min_price)
        else:
            df['price_normalized'] = df['price']
        
        max_volume = df['volume'].max()
        if max_volume > 0:
            df['volume_normalized'] = df['volume'] / max_volume
        else:
            df['volume_normalized'] = df['volume']
        
        # Временные признаки
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['weekday'] = df.index.weekday
        
        # RSI (Relative Strength Index)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Избегать деления на ноль
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Удаление строк с пропущенными значениями
        df = df.dropna()
        
        return df
    
    except Exception as e:
        logger.error(f"Ошибка предобработки данных: {str(e)}")
        return df

def resample_data(df, timeframe):
    """
    Ресэмплирование данных к определенному таймфрейму
    """
    try:
        # Ресэмплирование данных
        df_resampled = df.resample(timeframe).agg({
            'price': 'last',
            'volume': 'sum',
            'return': 'sum',
            'log_return': 'sum',
            'price_change_1': 'last',
            'price_change_6': 'last',
            'price_change_12': 'last',
            'price_change_24': 'last',
            'sma_5': 'last',
            'sma_10': 'last',
            'sma_20': 'last',
            'ema_5': 'last',
            'ema_10': 'last',
            'ema_20': 'last',
            'volume_sma_5': 'last',
            'volatility_5': 'last',
            'volatility_10': 'last',
            'price_normalized': 'last',
            'volume_normalized': 'mean',
            'rsi_14': 'last',
            'hour': 'first',
            'day': 'first',
            'month': 'first',
            'weekday': 'first'
        })
        
        # Удаление строк с пропущенными значениями
        df_resampled = df_resampled.dropna()
        
        logger.info(f"Выполнено ресэмплирование к {timeframe} с {len(df_resampled)} строками")
        return df_resampled
    
    except Exception as e:
        logger.error(f"Ошибка ресэмплирования данных: {str(e)}")
        return None

def create_lstm_sequences(df, sequence_length=24, prediction_horizons=[1, 3, 6, 12, 24]):
    """
    Создание последовательностей для обучения LSTM модели
    """
    try:
        # Проверка достаточности данных
        if len(df) < sequence_length + max(prediction_horizons):
            logger.warning(f"Недостаточно данных для LSTM последовательностей (нужно {sequence_length + max(prediction_horizons)}, есть {len(df)})")
            return None, None, None, None
        
        # Расчет будущих доходностей для горизонтов прогнозирования
        for horizon in prediction_horizons:
            df[f'future_return_{horizon}'] = df['price'].pct_change(periods=horizon, fill_method=None).shift(-horizon)
        
        # Удаление строк с пропущенными значениями в целевых столбцах
        target_cols = [f'future_return_{h}' for h in prediction_horizons]
        df = df.dropna(subset=target_cols)
        
        # Выбор признаков (топ-10 для соответствия нашей оптимизированной модели)
        feature_cols = ['price', 'volume', 'return', 'price_change_1', 
                       'sma_5', 'ema_5', 'volume_normalized', 'volatility_5',
                       'price_normalized', 'rsi_14']
        
        # Подготовка последовательностей
        X = []
        y = []
        
        for i in range(len(df) - sequence_length):
            X.append(df[feature_cols].iloc[i:i+sequence_length].values)
            y.append(df[target_cols].iloc[i+sequence_length-1].values)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Создано {len(X)} последовательностей с формой {X.shape}")
        return X, y, feature_cols, target_cols
    
    except Exception as e:
        logger.error(f"Ошибка создания LSTM последовательностей: {str(e)}")
        return None, None, None, None

def train_lstm_model(X_train, y_train, X_test, y_test, feature_cols, target_cols, 
                    hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001, 
                    batch_size=32, num_epochs=10, patience=5):
    """
    Обучение LSTM модели
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Обучение на устройстве: {device}")
        
        input_size = X_train.shape[2]  # Количество признаков
        output_size = y_train.shape[1]  # Количество целевых переменных
        
        # Создание модели
        model = PricePredictionLSTM(input_size, hidden_size, num_layers, output_size, dropout)
        model.to(device)
        
        # Оптимизатор и функция потерь
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Конвертация данных в тензоры
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        # Создание DataLoader для обучения
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Для отслеживания прогресса обучения
        train_losses = []
        test_losses = []
        
        # Для раннего останова
        best_test_loss = float('inf')
        no_improvement = 0
        best_model_state = None
        
        # Обучение модели
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Прямой проход
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Обратное распространение
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Проверка на валидационной выборке
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor).item()
                test_losses.append(test_loss)
            
            logger.info(f"Эпоха {epoch+1}/{num_epochs} - Потеря на обучении: {train_loss:.4f}, Потеря на тесте: {test_loss:.4f}")
            
            # Проверка для раннего останова
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = model.state_dict().copy()
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    logger.info(f"Ранний останов на эпохе {epoch+1}")
                    break
        
        # Загрузка лучшей модели
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Вычисление метрик на тестовой выборке
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()
            
            # Метрики для каждой целевой переменной
            rmse_values = [np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(output_size)]
            mae_values = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(output_size)]
            
            # Направленная точность (процент предсказаний с правильным знаком)
            direction_accuracy = []
            for i in range(output_size):
                correct_dir = np.sum(np.sign(y_pred[:, i]) == np.sign(y_true[:, i]))
                non_zero = np.sum(y_true[:, i] != 0)
                if non_zero > 0:
                    direction_accuracy.append(correct_dir / non_zero)
                else:
                    direction_accuracy.append(0.0)
        
        # Лучшие метрики тестирования
        best_test_metrics = {
            'rmse': {target_cols[i]: rmse_values[i] for i in range(output_size)},
            'mae': {target_cols[i]: mae_values[i] for i in range(output_size)},
            'direction_accuracy': {target_cols[i]: direction_accuracy[i] for i in range(output_size)},
            'avg_rmse': np.mean(rmse_values),
            'avg_mae': np.mean(mae_values),
            'avg_direction_accuracy': np.mean(direction_accuracy)
        }
        
        logger.info(f"Обучение завершено. Средний RMSE на тесте: {best_test_metrics['avg_rmse']:.4f}, Средний MAE на тесте: {best_test_metrics['avg_mae']:.4f}")
        logger.info(f"Средняя точность направления на тесте: {best_test_metrics['avg_direction_accuracy']:.4f}")
        
        return model, train_losses, test_losses, best_test_metrics
    
    except Exception as e:
        logger.error(f"Ошибка обучения LSTM модели: {str(e)}")
        return None, [], [], {}

def save_model(model, feature_cols, target_cols, metrics, model_dir, exchange, symbol):
    """
    Сохранение обученной модели
    """
    try:
        # Создание директории, если её нет
        os.makedirs(model_dir, exist_ok=True)
        
        # Путь для сохранения модели
        model_path = os.path.join(model_dir, f"lstm_{exchange}_{symbol}.pt")
        model_info_path = os.path.join(model_dir, f"lstm_{exchange}_{symbol}_info.json")
        
        # Сохранение модели
        torch.save(model.state_dict(), model_path)
        
        # Сохранение информации о модели
        model_info = {
            'name': f"{exchange}_{symbol}",
            'model_type': 'LSTM',
            'feature_columns': feature_cols,
            'target_columns': target_cols,
            'metrics': metrics,
            'input_size': len(feature_cols),
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'output_size': len(target_cols),
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        logger.info(f"Модель сохранена в {model_path}")
        return model_path
    
    except Exception as e:
        logger.error(f"Ошибка сохранения модели для {exchange}_{symbol}: {str(e)}")
        return None

def process_and_train(symbol, input_dir='attached_assets', model_dir='models', exchange='binance', sequence_length=24):
    """
    Обработка данных и обучение модели для одной криптовалютной пары
    """
    try:
        logger.info(f"Обработка и обучение для {symbol} из биржи {exchange}")
        
        # Путь к файлу данных
        filepath = os.path.join(input_dir, f"{symbol}.csv")
        if not os.path.exists(filepath):
            logger.error(f"Файл {filepath} не существует")
            return False
        
        # Загрузка данных
        df = load_csv(filepath)
        if df is None or len(df) == 0:
            logger.error(f"Не удалось загрузить данные для {symbol}")
            return False
        
        # Предобработка данных
        df = preprocess_data(df)
        if df is None or len(df) == 0:
            logger.error(f"Не удалось предобработать данные для {symbol}")
            return False
        
        # Ресэмплирование к часовому таймфрейму
        df_resampled = resample_data(df, '1h')
        if df_resampled is None or len(df_resampled) == 0:
            logger.error(f"Не удалось ресэмплировать данные для {symbol}")
            return False
        
        # Создание последовательностей для LSTM
        X, y, feature_cols, target_cols = create_lstm_sequences(df_resampled, sequence_length=sequence_length)
        if X is None or y is None or len(X) == 0:
            logger.error(f"Не удалось создать LSTM последовательности для {symbol}")
            return False
        
        # Разделение на обучающую и тестовую выборки (80/20)
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        logger.info(f"Обучающий набор: {X_train.shape}, Тестовый набор: {X_test.shape}")
        
        # Обучение модели
        model, train_losses, test_losses, metrics = train_lstm_model(
            X_train, y_train, X_test, y_test, feature_cols, target_cols, 
            hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001, 
            batch_size=32, num_epochs=10, patience=5
        )
        
        if model is None:
            logger.error(f"Не удалось обучить модель для {symbol}")
            return False
        
        # Сохранение модели
        model_path = save_model(model, feature_cols, target_cols, metrics, model_dir, exchange, symbol)
        if model_path is None:
            logger.error(f"Не удалось сохранить модель для {symbol}")
            return False
        
        # Построение графика обучения
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Потеря на обучении')
        plt.plot(test_losses, label='Потеря на тесте')
        plt.title(f'Кривые обучения для {exchange}_{symbol}')
        plt.xlabel('Эпоха')
        plt.ylabel('Потеря (MSE)')
        plt.legend()
        plot_dir = os.path.join(model_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"lstm_{exchange}_{symbol}_training.png"))
        plt.close()
        
        # Отображение метрик
        logger.info(f"Метрики для {exchange}_{symbol}:")
        logger.info(f"  Средний RMSE: {metrics['avg_rmse']:.4f}")
        logger.info(f"  Средний MAE: {metrics['avg_mae']:.4f}")
        logger.info(f"  Средняя точность направления: {metrics['avg_direction_accuracy']:.4f}")
        
        # Детальные метрики для временных горизонтов
        for i, target in enumerate(target_cols):
            logger.info(f"  {target}:")
            logger.info(f"    RMSE: {metrics['rmse'][target]:.4f}")
            logger.info(f"    MAE: {metrics['mae'][target]:.4f}")
            logger.info(f"    Точность направления: {metrics['direction_accuracy'][target]:.4f}")
        
        logger.info(f"Успешно обучена модель для {exchange}_{symbol}")
        return True
    
    except Exception as e:
        logger.error(f"Ошибка процесса обучения для {symbol}: {str(e)}")
        return False

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Обучение модели на одной паре криптовалют')
    parser.add_argument('symbol', type=str, help='Символ для обучения (без расширения .csv)')
    parser.add_argument('--input_dir', type=str, default='attached_assets', help='Директория с CSV файлами')
    parser.add_argument('--model_dir', type=str, default='models', help='Директория для сохранения моделей')
    parser.add_argument('--exchange', type=str, default='binance', help='Название биржи')
    parser.add_argument('--sequence_length', type=int, default=24, help='Длина последовательностей для LSTM')
    args = parser.parse_args()
    
    # Создание директории для моделей
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Обучение модели для указанного символа
    success = process_and_train(args.symbol, args.input_dir, args.model_dir, args.exchange, args.sequence_length)
    
    if success:
        logger.info(f"Обучение модели для {args.symbol} завершено успешно")
    else:
        logger.error(f"Обучение модели для {args.symbol} завершилось с ошибкой")

if __name__ == "__main__":
    main()