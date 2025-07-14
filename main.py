from app import app, db

if __name__ == "__main__":
    with app.app_context():
        # Make sure to import the models here so their tables will be created
        import models
        from settings import DEFAULT_SETTINGS
        
        # Initialize the database
        db.create_all()
        
        # Add default settings if not present
        for setting in DEFAULT_SETTINGS:
            existing = models.Setting.query.filter_by(key=setting['key']).first()
            if not existing:
                new_setting = models.Setting(
                    key=setting['key'],
                    value=setting['value'],
                    description=setting.get('description')
                )
                db.session.add(new_setting)
        
        db.session.commit()
        
    app.run(host="0.0.0.0", port=5000, debug=True)
