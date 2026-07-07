import os
from pathlib import Path

from flask import Flask

from .extensions import db

def create_app():
    app = Flask(__name__)
    base_dir = Path(app.root_path).parent
    upload_folder = base_dir / 'app' / 'static' / 'uploads'
    dataset_folder = base_dir / 'dataset' / 'augmented_balanced_dataset_zipped'
    database_url = os.environ.get('DATABASE_URL')

    if database_url:
        database_url = database_url.replace('postgres://', 'postgresql://', 1)

    app.config['UPLOAD_FOLDER'] = upload_folder.as_posix()
    app.config['DATASET_FOLDER'] = os.environ.get('PLANTINSIGHT_DATASET_FOLDER', dataset_folder.as_posix())
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url or f"sqlite:///{(base_dir / 'plantinsight.db').as_posix()}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True}

    upload_folder.mkdir(parents=True, exist_ok=True)

    db.init_app(app)

    from .routes import main
    from . import models  # noqa: F401

    app.register_blueprint(main)

    with app.app_context():
        db.create_all()

    return app