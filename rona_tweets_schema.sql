-- Schema For coronaTweets database

-- SCHEMAS FOR THE DATABASE
-- Drop Table if they exists
DROP TABLE coronaTweets;

-- CREATE TABLE

-- 1. coronaTweets
CREATE TABLE coronaTweets (

	id SERIAL,
	tweeet_date DATE NOT NULL,
	location VARCHAR(256) NOT NULL,
	tweet VARCHAR(256) NOT NULL,
	sentiment VARCHAR(256) NOT NULL,
	status VARCHAR(256) NOT NULL

);