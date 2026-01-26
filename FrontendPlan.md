# Page 1 - Model Training

[ ] Page load
- check api health on both port 8000 and 8001


## Section 1 Page One Data ingestion

- [ ] Upload CSV
    - [ ] name of CSV
    - [ ] Button to upload csv with name in textbox
    - [ ] will call the /new API
    - [ ] will call the /validate API
- [ ] Button to reassign training and testing data
    - [ ] name of CSV to reassign (Dropdown or tab maybe)
    - [ ] call the /validate API
- [ ] View training data
    - [ ] Graph or table with /training
- [ ] View Testing data
    - [ ] Graph or table with /testing
- [ ] View all data
    - [ ] /view to filter (3 above could be a filter component, or slicer in power bi)
- [ ] Clear database
    - [ ] dropdown with database names (database needs a refactor to change db name to table name but should not impact this development)
- [ ] Insert data
    - [ ] /insert no real place to use this now, maybe will be used for another API.


## Section 2 below ML model

- [ ] train model on specific dataset and model
    - [ ] /train
    - [ ] header should contain database name, maybe model name
    - [ ] type of model will be an option in the future if random forest isnt the only option we allow
    - [ ] progress bar of model training status/browser notification when the training is complete
- [ ] test model
    - [ ] /test or /model/metrics (i dont remember)
    - [ ] specific database name and model name
    - [ ] KPIs & graphs about the models
    - [ ] filter trained and untrained models with /model/status

# Page 2 - Live data and model classification

- [ ] Websocket with Kafka to predict data (mostly backend for now so on hold)
    - [ ] for now doing /predict on each of them will do
- [ ] filter data (location, user, ip, normal, malicious, usernames, comapny departments, countries, etc.)
- [ ] Also filter for before/after times of day 1h, 3h, 6h, 12h, 24h, 7d, 30d for both data and graphs
- [ ] refresh interval maybe to refresh graph
- [ ] look for bot headers, time on webpage, failed TLS handshakes & other indicators
- [ ] # active sessions
- [ ] globe with each of the user's active sessions
- [ ] graphs with the following KPIs
    - [ ] average session duration
    - [ ] requests per minute
    - [ ] total requests
    - [ ] allowed requests (with % allowed red if greater than ...%)
    - [ ] blocked requests
    (3 above in a single graph with each different colours)
    - [ ] types of threat
    - [ ] globe heat map
    - [ ] top 10 most blocked... users? countries? OS? Browser? ...
    - [ ] top 10 most attacked endpoints
        - [ ] # of requests for endpoint
        - [ ] % blocked for endpoint
    - [ ] Captcha solve rate
- [ ] compare multiple models at the same time on the same data (2 tables side by side and compare differences)
- [ ] last 10 blocked requests
- [ ] KPIs in fall report: accuracy, F1 score...
- [ ] on hold, users and their history (update to backend required)
    -...


# Page 3 - Sample app with auth

- [ ] Web page with username, password prompt
- [ ] Rate limiting, IP lists, time on webpage, biometrics, credential stuffing other tools to identify bots
- [ ] if malicious then block user
- [ ] options to enable options ie. captcha, tls, or something like that
- [ ] links impossible to reach, cookies, etc.