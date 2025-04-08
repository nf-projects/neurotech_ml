# Stimulations (target/nontarget)
https://openvibe.inria.fr/stimulation-codes/

target and nontarget = BOTH beeps

target = delayed
SimulationId_Labelstart = where delayed beep did NOT take place, but was "SUPPOSED" to take place

nontarget = non delayed (regular)

the ID's are:
OVTK_StimulationId_Label_01 (33025) = BUTTON PRESS
OVTK_StimulationId_NonTarget (33286) = NON DELAYED BEEP
OVTK_StimulationId_Target (33285) = DELAYED BEEP
OVTK_StimulationId_TrialStart (32773) = Trial BEGINS

# Libraries
NeuroKit2

# Processing
- Temporal Filter: Filter out frequencies
 -> Beta Waves: 12-30

- CSP (Spatial Filter):
  -> Find electrodes (dimensions) that are the strongest
- Epoching: Taking chunks of data out
  -> Time based
  -> stimulation based: during those time frames, take out a chunk of data

# Experiment Setup

Experiment was at 250Hz

10-20 typical EEG electrode placement

# Objective

Is there a diff. in delta and beta waves when there is a stimulus?

Mess around with epoch duration/offset