from flask import Flask,render_template,url_for, request, redirect
#from flask import Flask, render_template, request, redirect
from flaskext.mysql import MySQL

app=Flask(__name__)

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = '707@Gmail'
app.config['MYSQL_DATABASE_DB'] = 'Search_car'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

mysql = MySQL()
mysql.init_app(app)
conn = mysql.connect()
cursor = conn.cursor()

@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == "POST":


        CAR_INFO= request.form["CAR_INFO"]
        print(CAR_INFO)
        # # search
        sqlQuery = "SELECT * FROM CAR_INFO WHERE CarNumberPlateNo in ('"+CAR_INFO+"')";

        #sqlQuery = "SELECT * FROM CAR_INFO WHERE CarNumberPlateNo in ('DHAKA-METRO-GA-123456')";

        cursor.execute(sqlQuery)
                       
        conn.commit()
        data = cursor.fetchall()
        # # all in the search box will return all the tuples
        # if len(data) == 0 and CAR_INFO == 'all': 
        #     cursor.execute("SELECT CarNumberPlateNo, Location, TIME_DATE from CAR_INFO")
        #     conn.commit()
        #     data = cursor.fetchall()
        # data="razu"
        print(data)
        return render_template('search.html', data=data)
    return render_template('search.html')

if __name__ == "__main__":
    app.run(debug=True)


    