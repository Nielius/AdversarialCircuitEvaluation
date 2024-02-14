from acdc.nudb.adv_opt.data_fetchers import EXPERIMENT_DATA_PROVIDERS, AdvOptTaskName

# model = HookedTransformer.from_pretrained("solu-2l") # a very small model
#
# del model
#
# gc.collect()
#
tracr_reverse = EXPERIMENT_DATA_PROVIDERS[
    AdvOptTaskName.TRACR_REVERSE
].get_experiment_data()

model = tracr_reverse.masked_runner.masked_transformer.model

print(model)


from torchsummary import summary

help(summary)

x = model.state_dict()

summary(model, (1, 1000), device="cpu")
