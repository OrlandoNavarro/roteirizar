class DataService:
    def __init__(self, db_session):
        self.db_session = db_session

    def save_data(self, model_instance):
        try:
            self.db_session.add(model_instance)
            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            raise e

    def get_all_data(self, model):
        return self.db_session.query(model).all()

    def get_data_by_id(self, model, data_id):
        return self.db_session.query(model).filter_by(id=data_id).first()

    def update_data(self, model_instance):
        try:
            self.db_session.merge(model_instance)
            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            raise e

    def delete_data(self, model_instance):
        try:
            self.db_session.delete(model_instance)
            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            raise e