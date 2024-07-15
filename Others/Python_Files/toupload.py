import neptune
run = neptune.init_run(
    project="vishnumass/Churn-Analysis",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3NWM5MDg1Zi0yMjMwLTQ4NWYtOGYyMC00NDQ3NTYyMWM3OTEifQ==",
) 

run["result/output1"].upload(neptune.types.File.as_image("Confusion-Matrix_Logis.png"))
run["result/output2"].upload(neptune.types.File.as_image("Confusion-Matrix_Naive.png"))
run["result/output3"].upload(neptune.types.File.as_image("Confusion-Matrix_RandomForest.png"))
run["result/output4"].upload(neptune.types.File.as_image("Tree_RandomForest.png"))

run.stop()