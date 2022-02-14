from flask import Flask, render_template, request, send_file
from math import sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, pi, e, log
import sympy as sym
import numpy as np

def daoHam(func, xval):
  x = sym.symbols('x')
  f = str(sym.diff(func, x))
  x = xval
  return eval(f)

def daoHam2(func, xval):
  x = sym.symbols('x')
  f = str(sym.diff(sym.diff(func, x), x))
  x = xval
  return eval(f)

def chuanMot(arr):
  temp = [0]*len(arr[0])
  for i in range(len(arr)):
    for j in range(len(arr[0])):
      temp[j] += abs(arr[i][j])
  return max(temp)

def chuanVoCung(arr):
  temp = [0]*len(arr)
  for i in range(len(arr)):
    for j in range(len(arr[0])):
      temp[i] += abs(arr[i][j])
  return max(temp)

def subtract(arr1, arr2):
  arr = []
  for i in range(len(arr1)):
    temp = []
    for j in range(len(arr1[0])):
      x = arr1[i][j] - arr2[i][j]
      temp.append(x)
    arr.append(temp)
  return arr

def add(arr1, arr2):
  arr = []
  for i in range(len(arr1)):
    temp = []
    for j in range(len(arr1[0])):
      x = arr1[i][j] + arr2[i][j]
      temp.append(x)
    arr.append(temp)
  return arr

def jacobi(X, B, C, n):
  Xr = []
  Xr.append(X)

  T = []
  for i in range(len(B)):
    temp = [0]*len(B[0])
    for j in range(len(B[0])):
      if i == j:
        temp[j] = 0
      else:
        temp[j] = -B[i][j]/B[i][i]
    T.append(temp)

  for i in range(n):
    temp = []
    for j in range(len(T)):
      x = C[j][0]/B[j][j]
      for k in range(len(T[0])):
        x += T[j][k]*X[k][0]
      temp.append([x])
    X = temp
    Xr.append(X)

  Tn1 = ((chuanMot(T)**n)/(1-chuanMot(T)))*chuanMot(subtract(Xr[1], Xr[0]))
  Tni = ((chuanVoCung(T)**n)/(1-chuanVoCung(T)))*chuanVoCung(subtract(Xr[1], Xr[0]))
  Hn1 = (chuanMot(T)/(1-chuanMot(T)))*chuanMot(subtract(Xr[n-1], Xr[n-2]))
  Hni = (chuanVoCung(T)/(1-chuanVoCung(T)))*chuanVoCung(subtract(Xr[n-1], Xr[n-2]))
  ss = [Tn1, Tni, Hn1, Hni]
  return Xr, ss

def Gauss(X, B, C, n):
  Xr = []
  Xr.append(X)
  
  k = len(B)
  DL = np.array([[0]*k]*k)
  U = np.array([[0]*k]*k)
  for i in range(k):
    for j in range(k):
      if j < i+1:
        DL[i][j] = 1
        U[i][j] = 0
      else:
        DL[i][j] = 0
        U[i][j] = -1

  for i in range(k):
    for j in range(k):
      DL[i][j] = DL[i][j] * B[i][j]
      U[i][j] = U[i][j] * B[i][j]

  dl = np.linalg.inv(DL)
  T = np.dot(dl, U).tolist()

  for i in range(n):
    temp = []
    t = []
    for val in X:
      t.append(val[0])
    for j in range(len(B)):
      x = C[j][0]/B[j][j]
      for k in range(len(B[0])):
        if k != j:
          x -= (B[j][k]*t[k])/B[j][j]
      t[j] = x
      temp.append([x])
    X = temp
    Xr.append(X)
  
  Tn1 = ((chuanMot(T)**n)/(1-chuanMot(T)))*chuanMot(subtract(Xr[1], Xr[0]))
  Tni = ((chuanVoCung(T)**n)/(1-chuanVoCung(T)))*chuanVoCung(subtract(Xr[1], Xr[0]))
  Hn1 = (chuanMot(T)/(1-chuanMot(T)))*chuanMot(subtract(Xr[n-1], Xr[n-2]))
  Hni = (chuanVoCung(T)/(1-chuanVoCung(T)))*chuanVoCung(subtract(Xr[n-1], Xr[n-2]))
  ss = [Tn1, Tni, Hn1, Hni]

  return Xr, ss

def chiaDoi(func, a, b, n):
  res = []
  x = a
  if eval(func) < 0:
    isNormal = 1
  else:
    isNormal = 0
  for i in range(n+1):
    x = (a+b)/2
    f = eval(func)
    res.append([a, b, x, f, (b-a)/2])
    if isNormal == 1:
      if f < 0:
        a = x
      else:
        b = x
    else:
      if f < 0:
        b = x
      else:
        a = x
  return res
# print(chiaDoi("2.26*e**(-x) -2.89*x + cos(x+1)", -2, 2, 5))

def lapDon(func, a, b, x, q, n):
  res = []
  for i in range(n+1):
    xt = eval(func)
    ssh = q/(1-q) * (xt-x)
    res.append([x, xt, ssh])
    x = xt
  sst = (q**n)/(1-q) + (res[1][1]-res[1][0]) 
  return res, sst

def timHeSoCo(func, a, b):
  qa = daoHam(func, a)
  qb = daoHam(func, b)
  return max(qa, qb)

def dkFourier(func, a, b):
  x = 0
  f = daoHam(func, a)
  ff = daoHam2(func, a)
  if f*ff < 0:
    x = a
  else:
    x = b
  F = eval(func)
  ff = daoHam2(func, x)
  if F*ff > 0:
    isValid = 1
  else:
    isValid = 0
  return x, isValid

def findm(func, a, b):
  stoA = a
  stoX = []
  x = sym.symbols('x')
  f = str(sym.diff(func, x))
  while (a < b):
    x = a
    stoX.append(eval(f))
    a += 0.0001
  return stoA + stoX.index(min(stoX))*0.0001

def Newton(func, a, b, x, m, n):
  res = []
  for i in range(n+1):
    xo = x
    xt = x - (eval(func)/daoHam(func, x))
    x = xt
    ss = eval(func)/m
    res.append([xo, xt, ss])
  return res

app = Flask('__name__')
@app.route("/")
def home():
  return render_template('home.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
    
@app.route("/icon.png")
def icon():
  return send_file(r'templates/includes/icon.png', mimetype='image/png')

@app.route("/emoji.png")
def emoji():
  return send_file(r'templates/includes/emoji.png', mimetype='image/png')

@app.route("/newton", methods=['GET', 'POST'])
def newton():
  try:
    func = ""
    a = 0
    b = 0
    n = 0
    m = 0
    x = 0
    isValid = 0
    res = []
    if request.method == "POST":
      if int(request.form.get("n")) == 0:
        func = request.form.get("func")
        a = float(request.form.get("a"))
        b = float(request.form.get("b"))
        m = findm(func, a, b)
        x, isValid = dkFourier(func, a, b)
      else:
        func = request.form.get("func")
        a = float(request.form.get("a"))
        b = float(request.form.get("b"))
        m = float(request.form.get("m"))
        x = float(request.form.get("x"))
        n = int(request.form.get("n"))
        isValid = dkFourier(func, a, b)[1]
        if isValid == 1:
          res = Newton(func, a, b, x, m, n)
    if isValid == 1:
      dk = "OK"
    else:
      dk = "NO"
    return render_template('newton.html', func=func, a=a, b=b, n=n, x=x, m=m, res=res, dk=dk)
  except Exception as e:
    render_template('503.html')

@app.route("/lapdon", methods=['GET', 'POST'])
def lapdon():
  try:
    func = ""
    a = 0
    b = 0
    n = 0
    q = 0
    x = 0
    sst = 0
    res = []
    if request.method == "POST":
      if int(request.form.get("n")) == 0:
        func = request.form.get("func")
        a = float(request.form.get("a"))
        b = float(request.form.get("b"))
        q = timHeSoCo(func, a, b)
      else:
        func = request.form.get("func")
        a = float(request.form.get("a"))
        b = float(request.form.get("b"))
        q = float(request.form.get("q"))
        x = float(request.form.get("x"))
        n = int(request.form.get("n"))
        res, sst = lapDon(func, a, b, x, q, n)
    return render_template('lapdon.html', func=func, a=a, b=b, n=n, x=x, q=q, res=res, sst=sst)
  except Exception as e:
    render_template('503.html')

@app.route("/chiadoi", methods=['GET', 'POST'])
def chiadoi():
  try:
    func = ""
    a = 0
    b = 0
    n = 0
    res = []
    if request.method == "POST":
      func = request.form.get("func")
      a = float(request.form.get("a"))
      b = float(request.form.get("b"))
      n = int(request.form.get("n"))
      res = chiaDoi(func, a, b, n)
    return render_template('chiadoi.html', func=func, a=a, b=b, n=n, res=res)
  except Exception as e:
    render_template('503.html')
  
@app.route("/jacobi", methods=['GET', 'POST'])
def jacob():
  try:
    B = []
    C = []
    X = []
    n = 0
    Xr = []
    ss = [0]*4
    level = 0
    if request.method == "POST":
      level = int(request.form.get("level"))
      for i in range(level):
        t = []
        for j in range(level):
          t.append(0)
        B.append(t)
        C.append([0])
        X.append([0])
      if int(request.form.get("n")) == 0:
        pass
      else:
        for i in range(level):
          t = []
          for j in range(level):
            B[i][j] = float(request.form.get(str(i)+'-'+str(j)))
          C[i][0] = float(request.form.get(str(i)+'-C'))
          X[i][0] = float(request.form.get(str(i)+'-O'))
          n = int(request.form.get('n'))
        Xr, ss = jacobi(X, B, C, n)
    return render_template('jacobi.html', level=level, B=B, C=C, X=X, n=n, Xr=Xr, ss=ss)
  except Exception as e:
    render_template('503.html')

@app.route("/gauss", methods=['GET', 'POST'])
def gauss():
  try:
    B = []
    C = []
    X = []
    n = 0
    Xr = []
    ss = [0]*4
    level = 0
    if request.method == "POST":
      level = int(request.form.get("level"))
      for i in range(level):
        t = []
        for j in range(level):
          t.append(0)
        B.append(t)
        C.append([0])
        X.append([0])
      if int(request.form.get("n")) == 0:
        pass
      else:
        for i in range(level):
          t = []
          for j in range(level):
            B[i][j] = float(request.form.get(str(i)+'-'+str(j)))
          C[i][0] = float(request.form.get(str(i)+'-C'))
          X[i][0] = float(request.form.get(str(i)+'-O'))
          n = int(request.form.get('n'))
        Xr, ss = Gauss(X, B, C, n)
    return render_template('gauss.html', level=level, B=B, C=C, X=X, n=n, Xr=Xr, ss=ss)
  except Exception as e:
    render_template('503.html')

if __name__ == "__main__":
  app.debug = True
  app.run()
