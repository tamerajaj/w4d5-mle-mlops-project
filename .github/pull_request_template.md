## Checklist for the Refactoring Project

<!-- If you are done with a topic mark the checkboxes with an `x` (like `[x]`) -->

- [ ] I read and understood the tasks.
- [ ] I refactored the training notebook into a python file.
- [ ] I wrote a Github Action that trains the model using this python file.
- [ ] I added DVC to the Github Action.
- [ ] I deployed the API that uses the model.
- [ ] I deployed an evidently service, a prometheus service and grafana to GCP (or using the already deployed ones).
- [ ] I am monitoring the API and the model.


Bonus:
- [ ] I send requests to the API and monitor the data with evidently.
- [ ] I retrian the model with new data if I see data drift via the Github Action.