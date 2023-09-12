nohup python experiment.py -exp simpletcn_vloss_all >./simpletcn_vloss_all.txt &
nohup python experiment.py -exp mstcn_vloss >./mstcn_vloss.txt &
nohup python experiment.py -exp ms2_vloss >./ms2_vloss.txt &
nohup python experiment.py -exp asformer_vloss >./asformer_vloss.txt &
nohup python experiment.py -exp lstm_vloss_4_all >./lstm_vloss_4_all.txt &
nohup python experiment.py -exp gru_vloss_4_all >./gru_vloss_4_all.txt &
nohup python experiment.py -exp cldnn_vloss_4_all >./cldnn_vloss_4_all.txt &
nohup python experiment.py -exp bidirectional_vloss_4_all >./bidirectional_vloss_4_all.txt &


0904
nohup python experiment.py -exp lstm_vloss_1e4 >./lstm_vloss_1e4.txt &
nohup python experiment.py -exp gru_vloss_1e4 >./gru_vloss_1e4.txt &
nohup python experiment.py -exp cldnn_vloss_1e4 >./cldnn_vloss_1e4.txt &
nohup python experiment.py -exp bidirectional_vloss_1e4 >./bidirectional_vloss_1e4.txt &
nohup python experiment.py -exp simpletcn_vloss_1e4 >./simpletcn_vloss_1e4.txt &

# fft
nohup python experiment.py -exp lstm_vloss_1e4_fft -aug fft >./lstm_vloss_1e4_fft.txt &
nohup python experiment.py -exp gru_vloss_1e4_fft -aug fft >./gru_vloss_1e4_fft.txt &
nohup python experiment.py -exp cldnn_vloss_1e4_fft -aug fft >./cldnn_vloss_1e4_fft.txt &
nohup python experiment.py -exp bidirectional_vloss_1e4_fft -aug fft >./bidirectional_vloss_1e4_fft.txt &
nohup python experiment.py -exp simpletcn_vloss_1e4_fft -aug fft >./simpletcn_vloss_1e4_fft.txt &

# quantize
nohup python experiment.py -exp lstm_vloss_1e4_quantize -aug quantize >./lstm_vloss_1e4_quantize.txt &
nohup python experiment.py -exp gru_vloss_1e4_quantize -aug quantize >./gru_vloss_1e4_quantize.txt &
nohup python experiment.py -exp cldnn_vloss_1e4_quantize -aug quantize >./cldnn_vloss_1e4_quantize.txt &
nohup python experiment.py -exp bidirectional_vloss_1e4_quantize -aug quantize >./bidirectional_vloss_1e4_quantize.txt &
nohup python experiment.py -exp simpletcn_vloss_1e4_quantize -aug quantize >./simpletcn_vloss_1e4_quantize.txt &

# drift
nohup python experiment.py -exp lstm_vloss_1e4_drift_05 -aug drift >./lstm_vloss_1e4_drift_05.txt &
nohup python experiment.py -exp gru_vloss_1e4_drift_05 -aug drift >./gru_vloss_1e4_drift_05.txt &
nohup python experiment.py -exp cldnn_vloss_1e4_drift_05 -aug drift >./cldnn_vloss_1e4_drift_05.txt &
nohup python experiment.py -exp bidirectional_vloss_1e4_drift_05 -aug drift >./bidirectional_vloss_1e4_drift_05.txt &
nohup python experiment.py -exp simpletcn_vloss_1e4_drift_05 -aug drift >./simpletcn_vloss_1e4_drift_05.txt &

# timewrap
nohup python experiment.py -exp lstm_vloss_1e4_timewrap -aug timewrap >./lstm_vloss_1e4_timewrap.txt &
nohup python experiment.py -exp gru_vloss_1e4_timewrap -aug timewrap >./gru_vloss_1e4_timewrap.txt &
nohup python experiment.py -exp cldnn_vloss_1e4_timewrap -aug timewrap >./cldnn_vloss_1e4_timewrap.txt &
nohup python experiment.py -exp bidirectional_vloss_1e4_timewrap -aug timewrap >./bidirectional_vloss_1e4_timewrap.txt &
nohup python experiment.py -exp simpletcn_vloss_1e4_timewrap -aug timewrap >./simpletcn_vloss_1e4_timewrap.txt &