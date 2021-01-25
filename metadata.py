import sqlalchemy as sql
import sqlalchemy.ext.declarative
import sqlalchemy.orm
 
import uuid
import os
import pickle

Base = sql.ext.declarative.declarative_base()

class Model(Base):
    __tablename__ = 'model'
    id =         sql.Column(sql.Integer, primary_key=True)
    uuid   =     sql.Column(sql.Text,    nullable=False, index=True)
    dataset =    sql.Column(sql.Text,    nullable=False, index=True)
    k =          sql.Column(sql.Integer, nullable=False)
    klass =      sql.Column(sql.Text,    nullable=False)
    params_ =    sql.Column(sql.Text,    nullable=False)
    iterations = sql.Column(sql.Integer)
    duration =   sql.Column(sql.Float)
    perplexity = sql.Column(sql.Float)
    c_v =        sql.Column(sql.Float)
    c_umass =    sql.Column(sql.Float)
    c_npmi =     sql.Column(sql.Float)

    def __init__(self, **kwargs):
        self.uuid = str(uuid.uuid4())
        self.params_ = pickle.dumps({})
        for k,v in kwargs.items():
            self.__setattr__(k,v)
    
    @property
    def path(self):
        return 'models/%s.model' % self.uuid
    
    @property
    def params(self):
        return pickle.loads(self.params_)
    
    @params.setter
    def params(self, value):
        self.params_ = pickle.dumps(value)

# standard decorator style
@sql.event.listens_for(Model, 'after_delete')
def delete_persisted_model(mapper, connection, target):
    print('removing path "%s"' % target.path)
    try:
        os.remove(target.path)
    except FileNotFoundError: pass

# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.
engine = sql.create_engine('sqlite:///models.db')
 
# Create all tables in the engine. This is equivalent to "Create Table"
# statements in raw SQL.
Base.metadata.create_all(engine)

session = sql.orm.sessionmaker(bind=engine)()