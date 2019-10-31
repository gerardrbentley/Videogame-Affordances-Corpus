import socket
import time
import os

from flask_script import Manager


from db import drop_all, init_db, ingest_filesystem_data
from init import create_app

# sets up the app
app = create_app()
#db =
#init_app(app)

manager = Manager(app)
#migrate = Migrate(app, db)

# adds the python manage.py db init, db migrate, db upgrade commands
# manager.add_command("db", MigrateCommand)


@manager.command
def runserver():
    app.run(debug=True, host="0.0.0.0", port=5000)


@manager.command
def runworker():
    app.run(debug=True)


@manager.command
def ingest_files():
    print('CREATING TABLES IF NOT EXIST')
    init_db()
    print('INGESTING DATA FROM ../games (dockerroot/games)')
    ingest_filesystem_data('../games')  # ('../../affordances_corpus/games/')
    # app.run(debug=True, host='0.0.0.0', port=5000)


@manager.command
def recreate_db():
    """
    Recreates a database. This should only be used once
    when there's a new database instance. This shouldn't be
    used when you migrate your database.
    """

    drop_all()
    init_db()


if __name__ == "__main__":
    manager.run()
