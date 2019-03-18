# coding:utf-8
from flask_wtf import Form
from wtforms import TextAreaField,StringField, BooleanField
from wtforms.validators import DataRequired

class LoginForm(Form):
    openid = TextAreaField('openid')