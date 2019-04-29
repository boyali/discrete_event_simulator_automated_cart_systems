# Discrete Event Simulator and Supervisory Control by Reinforcement Learning for Autonomous Carts
This simulation environment is designed for simulating autonomous cart and design supervisory control with RL algorithms. The autonomous carts are managed by the control algorithm. Here we provide a simulation framework which can be used to analyze passenger waiting statistics. The RL study codes will be published soon separately. 

We provided a sample notebook file for statistical analysis in the repository. For simulation, please just run the 
standard_SIM_ord in tests folder. The related logs are generated and saved to be used with panda dataframes. 

For details of the simulation environment please check the technical report. For citations;

@article{hashimoto2018stochastic,
  title={Stochastic Discrete Event Simulation Environment for Autonomous Cart Fleet for Artificial Intelligent Training and Reinforcement Learning Algorithms (マルチメディアストレージ ヒューマンインフォメーション メディア工学 映像表現 \& コンピュータグラフィックス)},
  author={HASHIMOTO, Naohisa and BOYALI, Ali and KATO, Shin and OTSUKA, Takao and MIZUSHIMA, Kazuhisa and OMAE, Manabu},
  journal={電子情報通信学会技術研究報告= IEICE technical report: 信学技報},
  volume={117},
  number={431},
  pages={29--33},
  year={2018},
  publisher={電子情報通信学会}
}

We build training environment using Tensorflow. A sample training network is given in the network folder. One can run
 TF_train.py in the test folder to train using a couple of policies given in the script. 
 
For citations of the Supervisory Reinforcement Learning;
 
 Multi-Agent Reinforcement Learning for Autonomous On Demand Vehicles
 Ali Boyali, Naohisa Hashimoto, Vijay John and Tankut Acarman 
 Intelligent Vehicle Symposium IV 2019 (To be published)
