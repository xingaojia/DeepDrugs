import torch
import os

old_path = "0_fold_Drugcomb_best_model.pth"
new_path = "0_fold_Drugcomb_best_model.pth"

state_dict = torch.load(old_path, map_location='cpu')

if torch.cuda.is_available():
    new_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            new_state_dict[k] = v.to('cuda:0')
        else:
            new_state_dict[k] = v
else:
    print("⚠ NO GPU，saved at CPU")
    new_state_dict = state_dict

torch.save(new_state_dict, new_path)

print("✅ Finished!")
print(f"New model saved at: {new_path}")

