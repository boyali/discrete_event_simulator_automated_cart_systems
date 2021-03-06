# Discrete Event Simulator and Supervisory Control by Reinforcement Learning for Autonomous Carts
This simulation environment is designed for simulating autonomous cart and design supervisory control with RL algorithms. The autonomous carts are managed by the control algorithm. Here we provide a simulation framework which can be used to analyze passenger waiting statistics. The RL study codes will be published soon separately. 

We provided a sample notebook file for statistical analysis in the repository. For simulation, please just run the 
standard_SIM_ord in tests folder. The related logs are generated and saved to be used with panda dataframes. 

For details of the simulation environment please check the technical report. For citations;

    @article{hashimoto2018stochastic, <br>
      title={Stochastic Discrete Event Simulation Environment for Autonomous Cart Fleet for Artificial Intelligent Training and Reinforcement Learning Algorithms (マルチメディアストレージ ヒューマンインフォメーション メディア工学 映像表現 \& コンピュータグラフィックス)},<br>
      author={HASHIMOTO, Naohisa and BOYALI, Ali and KATO, Shin and OTSUKA, Takao and MIZUSHIMA, Kazuhisa and OMAE, Manabu},<br>
      journal={電子情報通信学会技術研究報告= IEICE technical report: 信学技報},<br>
      volume={117},<br>
      number={431},<br>
      pages={29--33},<br>
      year={2018},<br>
      publisher={電子情報通信学会}<br>
    }

[Stochastic Discrete Event Simulation Environment for Autonomous Cart Fleet for Artificial Intelligent Training and Reinforcement Learning Algorithms ](http://www.academia.edu/download/58863306/discrete_event_simulator_report_paper20190411-19568-t6qbrx.pdf)


We build training environment using Tensorflow. A sample training network is given in the network folder. One can run
 TF_train.py in the test folder to train using a couple of policies given in the script. 
 
For citations of the Supervisory Reinforcement Learning;
 
         @inproceedings{boyali2019multi,<br>
          title={Multi-Agent Reinforcement Learning for Autonomous On Demand Vehicles},<br>
          author={Boyal{\i}, Ali and Hashimoto, Naohisa and John, Vijay and Acarman, Tankut},<br>
          booktitle={2019 IEEE Intelligent Vehicles Symposium (IV)},<br>
          pages={1461--1468},<br>
          year={2019},<br>
          organization={IEEE}<br>
        }
        
[Multi-Agent Reinforcement Learning for Autonomous On Demand Vehicles](https://www.researchgate.net/profile/Ali_Boyali/publication/332446506_Multi-Agent_Reinforcement_Learning_for_Autonomous_On_Demand_Vehicles/links/5ceaa035299bf14d95bc20e5/Multi-Agent-Reinforcement-Learning-for-Autonomous-On-Demand-Vehicles.pdf)        
        
