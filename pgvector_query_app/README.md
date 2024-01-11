# PGVector Query API

This is a minimal API to query the VectorDB from outside the cluster(e.g from the `find-courses` command in [rhct-tools](https://github.com/RedHatTraining/curriculum-tools/tree/rhct)).

The vector database is deployed at: https://console-openshift-console.apps.rhods-internal.61tk.p1.openshiftapps.com/k8s/ns/rht-chatbot/deployments/postgresql

For instructions to populate this database with course content, see `examples/notebooks/langchain/Langchain-PgVector-Ingest.ipynb`.
