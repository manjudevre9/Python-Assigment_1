#Que.1.Explain GET & POST methods.
#Ans.
# GET Method:
# Definition:The GET method is used to retrieve data from a server.It requests a specific resource from the server,
# & the server responds with the requested data.

# Characteristics:
# 1.Retrieve data:GET is used to retrieve data from a server.
# 2.Data in URL:Data is passed in the URL as query parameters.
# 3.Cacheable:GET requests are cacheable,meaning the browser can store the responsed for furure requests.

# Use Cases:
# 1.Retrieve a web page.
# 2.Fetch data from an API.

# POST Method:
# Defination:The POST method is used to send data to a server to create or update a resource.

# Characteristics:
# 1.Send data:POST is used to send data to a server.
# 2.Data in request body:Data is passed in the request body,not in the URl.
# 3.Not cacheable:POST request are not cacheable.

# Use Cases:
# 1.Create a new user account.
# 2.Submit a form.
# 3.upload a file.

#Que.2.Why is request used in Flask?
# Ans.In Flask, the request object is a global object that provides access to the current HTTP request it is used to retrieve data from the HTTP request,such as:
# The request object,method used in Flask views to:
# 1.Retrieve user input:Get data from forms,query parameters,or JSON data.
# 2.Validate user input:Check the data for errors or inconsistencies.
# 3.Perform actions:Use the data to perform actions,such as creating or updating records in a database.

#Que.3.Why is redirect()used in Flask?
# Ans.In Flask,the redirect() function is used to redirect from one URL to another.The redirect() function 
# function takes a URL or a URL endpoint as an argument & returns a response object that redirects the 
# user to the specified URL.

#Que.4.What are templates in Flask? Why is the render_template()function used?
# Ans.In Flask, templates are HTML files that contain pleceholders for dynamic data.They allow you to separated
# the presentation layer of your application from the logic layer.

#render_template()function:The render_template()function is a built-in flask function that renders a
# template with the given variables.It takes two arguments.
# 1.Template name:The name of the file to render.
# 2.A dictionary of variable to pass the tamplate.

#5.CReate a simple API.Use Postman to test it.
# Ans.
from flask import Flask,jsonify,request

app=Flask(__name__)

books=[{"id":1,"title":"Book 1","author":"Author 1"},{"id":2,"title":"Book 2","author":"Author 2"},{"id":3,"title":"Book 3","author":"Author 3"}]
@app.route("/books",methods=["GET"])
def get_all_books():
  return jsonify(books)

@app.route("/books/<int:book_id>",methods=["GET"])
def get_book(book_id):
  for book in books:
   if book["id"]==book_id:
    return jsonify(book)
   return jsonify({"message":"Book not found"})

@app.route("/books",methods=["POST"])
def create_book():
   new_book={"id":len(books)+1,"title":request.json["title"],"another":request.jons["another"]}
   books.append(new_book)
   return jsonify(new_book),201

@app.route("/books/<int:book_id>",methods=["PUT"])
def udate_book(book_id):
   for book in books:
     if book["id"]==book_id:
       book["title"]=request.json.get("title",book["title"])
       book["author"]=request.json.get("author",book["author"])
       return jsonify(book)
     return jsonify({"message":"book not found"})

@app.route("/books/<int:book_id>", methods=["DELETE"])
def delete_book(book_id):
  for book in books:
    if book["id"]==book_id:
      books.remove(book)
      return jsonify({"message":"Book deleted"}) 
    return jsonify({"message":"Book not found"})

if__name__ == "__main__":
app.run(debug=True)


