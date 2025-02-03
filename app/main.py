from flask import Flask, request, render_template, redirect, url_for, flash
import settings
import utils
import numpy as np
import cv2
import Final_Prediction as pred
from models import app, db, User
from form import RegisterForm, LoginForm
from flask_login import login_user, logout_user, login_required, current_user

docscan = utils.DocumentScan()

@app.route('/')
@app.route('/home')
def home():
    return redirect(url_for('login_page'))

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(
            username=form.username.data,
            email_address=form.email_address.data,
            password=form.password1.data
        )
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        flash(f"Account created successfully! You are now logged in as {user_to_create.username}", category='success')
        return redirect(url_for('scandoc'))

    if form.errors:
        for error_list in form.errors.values():
            for error in error_list:
                flash(f"Error: {error}", category='danger')

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_password_correction(attempted_password=form.password.data):
            login_user(attempted_user)
            flash(f"Success! You are logged in: {attempted_user.username}", category='success')
            return redirect(url_for('scandoc'))
        else:
            flash("Username and password do not match! Please try again.", category='danger')

    return render_template('login.html', form=form)

@app.route('/log_out')
def log_out_page():
    logout_user()
    flash("You have been logged out!", category='info')
    return redirect(url_for('login_page'))

@app.route('/scandoc', methods=['GET', 'POST'])
@login_required
def scandoc():
    if request.method == 'POST':
        file = request.files['image_name']
        upload_image_path = utils.save_upload_image(file)
        print(f"Image saved in = {upload_image_path} ")

        # predict the coordinates of the document
        four_points, size = docscan.document_scanner(upload_image_path)
        print(four_points, size)
        if four_points is None:
            message = "UNABLE TO LOCATE THE COORDINATES OF DOCUMENT: points displayed are random"
            points = [
                {'x': 10, 'y': 10},
                {'x': 120, 'y': 10},
                {'x': 120, 'y': 120},
                {'x': 10, 'y': 120}
            ]
            return render_template('scanner.html',
                                   points=points,
                                   fileupload=True,
                                   message=message)
        else:
            points = utils.array_to_json_format(four_points)
            message = 'Located the Coordinates of Document using OpenCV'

            return render_template('scanner.html',
                                   points=points,
                                   fileupload=True,
                                   message=message)

    return render_template('scanner.html')

@app.route('/transform', methods=['POST'])
def transform():
    try:
        points = request.json['data']
        array = np.array(points)
        magic_color = docscan.calibrate_to_original_size(array)
        file_name = 'magic_color.jpg'
        magic_image_path = settings.join_path(settings.MEDIA_DIR, file_name)
        cv2.imwrite(magic_image_path, magic_color)

        return 'success'
    except Exception as e:
        print(f"Error in transform route: {e}")
        return 'fail'

@app.route('/prediction')
def prediction():
    wrap_image_filepath = settings.join_path(settings.MEDIA_DIR, 'magic_color.jpg')
    image = cv2.imread(wrap_image_filepath)
    image_bb, results = pred.get_predictions(image)
    results = results if results else {}

    bb_filename = settings.join_path(settings.MEDIA_DIR, 'bounding_box.jpg')
    cv2.imwrite(bb_filename, image_bb)
    return render_template('predictions.html', results=results)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Ensure the tables are created
    app.run(debug=True)
