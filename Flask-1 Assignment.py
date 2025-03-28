#Que 1. What is Flask framework?What are the advantages of flask framework?
#Ans.Flask is a popular Python web framework that allows developers to build web applications quickly & efficiently.
# It is a micriframework,meaning it has a lightweight & flexible architecture that makes it ideal for building 
# small to medium-sized web applications.
# Advantages:
# 1.Lightweight & Flexible.
# 2.Easy to learn.
# 3.Fast Development.
# 4.Extensive Libraries & Tools.
# 5.Large Community.
# 6.RESTful API Support.

#Que.2.Create a simple flask application to display "Hello World!!".
#Ans.
from flask import Flask

app=Flask(__name__)

@app.route('/')
def index():
    return "Hello World!!"
if __name__ == "__main__":
    app.run(debug=True)  

#Que.3.What is Approuting in Flask? Why do we use app routs?    
#Ans.In Flask,app routing refers to the process of mapping URLs to specific application endpoints.Its a way to 
# associate a URL with a particular function or view that handles the request.
# App routes are used to:
# 1.Map URLs to views:App routes map URLs to specific views or fubctions that handle the request.
# 2.Handle HTTP requests:App routs handle HTTP requests,such as GET,POST,PUT, & DELETE.
# 3. Organize application logic:App routes help organize application logic by seprating concerns & making it easier to maintain & scale the application.

#Que.4. Create a "/Welcome" route to display the welcome message "Welcome to ABC Corporation"& a"/"route to show the following details:
# Company name ABC corporation
# Location:India
# Contact details:999-999-9999
#Ans.
from flask import Flask
app=Flask(__name__)

@app.route('/welcome')
def wecome():
    return 'Welcome to ABC Corporation'
@app.route('/')
def company_details():
    company_name='ABC Corporation'
    location='India'
    contact_details='999-999-9999'

    details=f'Company Name:{company_name}\nLocation:{location}\nContact Details:{contact_details}'
    return details
if __name__ == "__main__":app.run(debug=True)

#Que.What function is used in Flask for URL Building?Write a python code to demonstrate the working of the url_for()function.
#Ans.In Flask,the url_for() function is used for URL building.This function generates URLs for specific routes in your application.
from flask import Flask, url_for
app=Flask(__name__)
@app.route('/')
def index():
    return 'Index page'
@app.route('/about')
def about():
    return 'About page'
@app.route('/contact')
def contact():
    return 'Contact page'
@app.route('/urls')
def urls():
    index_url=url_for('index')
    about_url=url_for('about')
    contact_url=url_for('contact')

    return f'Index URL:{index_url}\nAbout URL:{about_url}\nContact URL:{contact_url}'
if __name__ == "__main__":
    app.run(debug=True)

