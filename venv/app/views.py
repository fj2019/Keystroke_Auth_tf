# coding:utf-8
from flask import render_template, flash, redirect
from app import app
from .forms import LoginForm
from test import keycol_train
from test import keycol
from model_test import model_test
@app.route('/colloct', methods = ['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        k = keycol_train()
        k.main()
    return render_template('colloct.html',
        title='Detect Data',
        form = form)
@app.route('/test', methods = ['GET', 'POST'])
def login1():
    
    form = LoginForm()

    if form.validate_on_submit():

        k = keycol()
        k.main()
    return render_template('test.html',
        title='Detect Data',
        form = form)