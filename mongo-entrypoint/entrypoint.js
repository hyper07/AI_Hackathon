var db = connect("mongodb://admin:PassW0rd@localhost:27017/admin");

db = db.getSiblingDB('db');

db.createUser({
  user: "user",
  pwd: "pass",
  roles: [{
    role: "readWrite",
    db: "db"
  }],
  passwordDigestor: "server",
})