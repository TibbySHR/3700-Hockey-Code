import wandb

GAME_ID_TARGET = 2017021065
subset = shots[shots["game_id"] == GAME_ID_TARGET].copy()

# 可选：本地也存一份，便于核对
subset_out = "data/processed/wpg_v_wsh_2017021065.csv"
subset.to_csv(subset_out, index=False)
print(f"✅ Saved subset: {subset_out} (rows={len(subset)})")

# 上传到 WandB（把 project/name 换成你的）
run = wandb.init(project="ift6758-stage2", job_type="dataset-upload")

artifact_name = "wpg_v_wsh_2017021065"
artifact = wandb.Artifact(artifact_name, type="dataset")

# 用 Table 包装 DataFrame
table = wandb.Table(dataframe=subset)
artifact.add(table, artifact_name)

run.log_artifact(artifact)
run.finish()
print(f"✅ Logged W&B artifact: {artifact_name}")