using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using Algorithms.RL;
using Graphs;
using Gym;
using NN;
using UnityEngine;
using UnityEngine.UI;
using Random = System.Random;

namespace TestGround
{
    public class TestDQN : MonoBehaviour
    {
        [SerializeField] protected ComputeShader shader;
        [SerializeField] private WindowGraph windowGraphPrefab;
        [SerializeField] protected float simulationSpeed;
        [SerializeField] private int numberOfEpisodes;
        [SerializeField] private int targetNetworkCopyPeriod;
        private int _episodeIndex;
        protected float[] _currentSate;
        private int _totalIteration;
        protected StealthGameEnv _env;
        private List<float> _rewardsOverTime;
        private List<float> _lossPerEpisode;

        // DQN specific variables
        protected ModelDQN _DQN;
        protected float _epsilon;

        private WindowGraph _graphReward;
        private WindowGraph _graphLoss;

        private void Awake()
        {
            _env = FindObjectOfType<StealthGameEnv>();
            _rewardsOverTime = new List<float>(numberOfEpisodes);
            _lossPerEpisode = new List<float>(numberOfEpisodes);
            for (int i = 0; i < _rewardsOverTime.Capacity; i++)
            {
                _rewardsOverTime.Add(0f);
                _lossPerEpisode.Add(0f);
            }
        }

        protected virtual void Start()
        {
            _currentSate = _env.ResetEnv();

            var updateLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.Tanh, Instantiate(shader), true),
                new NetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(128, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader))
            };
            var updateModel = new NetworkModel(updateLayers, new MeanSquaredError(Instantiate(shader)));

            var targetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, 128, ActivationFunction.Tanh, Instantiate(shader), true),
                new NetworkLayer(128, 128, ActivationFunction.Tanh, Instantiate(shader)),
                new NetworkLayer(128, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader))
            };
            var targetModel = new NetworkModel(targetLayers, new MeanSquaredError(Instantiate(shader)));

            _DQN = new ModelDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }

        private void FixedUpdate()
        {
            if (_episodeIndex >= numberOfEpisodes)
            {
                if (!_env) return;

                _env.Close();
                PlotTrainingData();
                return;
            }

            int action = _DQN.EpsilonGreedySample(_currentSate, _epsilon);
            var stepInfo = _env.Step(action);

            _DQN.AddExperience(_currentSate, action, stepInfo.Reward, stepInfo.Done, stepInfo.Observation);
            _DQN.Train();

            if (_totalIteration % targetNetworkCopyPeriod == 0)
            {
                _DQN.SetTargetModel();
            }

            _currentSate = stepInfo.Observation;
            _rewardsOverTime[_episodeIndex] += stepInfo.Reward;
            ++_totalIteration;

            if (!stepInfo.Done) return;

            _lossPerEpisode[_episodeIndex] = _DQN.SampleLoss();
            _currentSate = _env.ResetEnv();
            ++_episodeIndex;

            // could clamp epsilon value with a max function in order to have a min exploration value
            _epsilon = 1.0f / (float)Math.Sqrt(_episodeIndex + 1);
        }

        private void PlotTrainingData()
        {
            Time.timeScale = 1;

            float rewardSum = 0.0f;
            foreach (var reward in _rewardsOverTime)
            {
                rewardSum += reward;
            }

            float lossSum = 0.0f;
            for (int i = 0; i < _lossPerEpisode.Count; i++)
            {
                var loss = _lossPerEpisode[i];
                lossSum += loss;
                // if (loss > 1)
                // {
                //     _lossPerEpisode[i] = 1.0f;
                // }
            }

            print("Average Reward: " + rewardSum / _rewardsOverTime.Count);
            print("Average Loss: " + lossSum / _lossPerEpisode.Count);

            var layoutGroup = FindObjectOfType<VerticalLayoutGroup>();
            _graphReward = Instantiate(windowGraphPrefab, layoutGroup.transform);
            _graphLoss = Instantiate(windowGraphPrefab, layoutGroup.transform);
            StartCoroutine(ShowGraphs());
        }

        private IEnumerator ShowGraphs()
        {
            yield return new WaitForSeconds(0.1f);

            _graphReward.SetGraph(null, _rewardsOverTime, GraphType.LineGraph,
                "Rewards per Episode", "episodes", "rewards");

            _graphLoss.SetGraph(null, _lossPerEpisode, GraphType.LineGraph,
                "Loss per Episode", "episodes", "loss");
        }

        private void OnDestroy()
        {
            Time.timeScale = 1;
            _DQN?.Dispose();
        }
    }
}