from flask import Flask, render_template, request, redirect, url_for
import datetime
app = Flask(__name__)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# Importing our classes from database_setup
from database_setup import Base, Annotation, Sprite_info, Tile, Tile_info, Sprite, Image_info

#Connect to Database and create database session
engine = create_engine('sqlite:///vgac-test.db')
# Bind the engine to the metadata of the Base class so that the
# declaratives can be accessed through a DBSession instance
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
# A DBSession() instance establishes all conversations with the database
# and represents a "staging zone" for all the objects loaded into the
# database session object.
session = DBSession()

@app.route('/')
@app.route('/annotations')
def showAnnotations():
   annotations = session.query(Annotation).all()
   return render_template("annotations.html", annotations=annotations)

 #This will let us Create a new tag and save it in our database
@app.route('/annotations/new/',methods=['GET','POST'])
def newAnnotation():
   if request.method == 'POST':
       newAnnotation = Annotation(img_id = request.form['img_id'], tagger_id = request.form['tagger_id'], affordance = int(request.form['affordance']), tags = bytes([int(request.form['tags'])]), created_at = datetime.datetime.now())
       session.add(newAnnotation)
       session.commit()
       return redirect(url_for('showAnnotations'))
   else:
       return render_template('newAnnotation.html')

if __name__ == '__main__':
   app.debug = True
   app.run(host='0.0.0.0', port=4996)
