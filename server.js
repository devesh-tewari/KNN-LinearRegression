var express = require('express');
var app = express();
var os = require('os');
var bodyParser = require('body-parser');
var passport = require('passport');
var LocalStrategy = require('passport-local').Strategy;
var cookieParser = require('cookie-parser');
var session = require('express-session');

app.set('port', (process.env.PORT || 5000));

//Mongoose database
var mongo = require('mongodb');
var monk = require('monk');
var db = monk(process.env.MONGOLAB_URI);

//Configuration
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(session({secret: 'townhall', resave: true, saveUninitialized: true }));

require('./config/passport')(app);


//DB
db.on('error', function err(){
    console.log('Error connecting to DB');
});
db.on('open', function callback() {
    console.log('Fresh DB Connected');
});

// Make our db accessible to our router
app.use(function (req, res, next) {
    req.db = db;
    next();
});

//CORS resolution
app.use(function (request, response, next) {

    response.header('Access-Control-Allow-Origin', '*');
    response.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    response.header('Access-Control-Allow-Headers', 'Origin, Content-Type');
    next();
});

//defining routes
var playerRouting = require('./routes/player-routes');
var userRouting = require('./routes/user-routes');
var coachRouting = require('./routes/coach-routes');
var atRouting = require('./routes/at-routes');

var userRouting = require('./auth');

app.use('/', express.static('client'));
app.use(express.static('client'));

//implementing routes
app.use('/player', playerRouting);
app.use('/user', userRouting);
app.use('/coach', coachRouting);
app.use('/atc', atRouting);
app.use('/signin', userRouting);

app.post('/signin', passport.auth);

app.listen(app.get('port'), function() {
  console.log('Node app is running on port', app.get('port'));
});
