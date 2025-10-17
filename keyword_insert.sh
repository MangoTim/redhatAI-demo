#!/bin/bash
keyword="$1"
response="$2"

psql -U rhai -h 192.168.147.103 -d rhai_table \
  -c "INSERT INTO keyword_responses (keyword, response) VALUES ('$keyword', '$response');"
