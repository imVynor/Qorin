# Break and Interruption Management Schedule

## Mandatory 1:45 / 15 Rule Implementation
- **Active Window**: 105 minutes (1 hour 45 minutes) continuous GPU training.
- **Cooling Window**: 15 minutes complete GPU idle.
- **Pre-Cooldown Action**: Complete active optimizer step, flush CUDA cache, write atomic checkpoint (`ckpt_latest.pt`).
- **Cooldown Behavior**: Free GPU memory, zero forward/backward passes, monitor temperature decline until GPU temp falls below 65°C.
- **Thermal Safety Override**: If GPU temperature reaches or exceeds 92°C at any step, training pauses immediately, checkpoints, and enters cooldown.
- **Windows Power Policy**: Disallow system sleep while plugged in; screen timeout allowed.
