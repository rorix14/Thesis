using System.Collections;
using System.Collections.Generic;
using Algorithms.NE;
using Graphs;
using Gym;
using NN;
using UnityEngine;
using UnityEngine.UI;

namespace TestGround.NE
{
    public class TestGA : MonoBehaviour
    {
        [SerializeField] private int populationSize;
        [SerializeField] private int eliteNumber;
        [SerializeField] private int tournamentNumber;
        [SerializeField] private float mutationNoiseStd;

        [SerializeField] private ComputeShader shader;
        [SerializeField] private WindowGraph windowGraphPrefab;
        [SerializeField] private float simulationSpeed;
        [SerializeField] private int numberOfEpisodes;

        private int _episodeIndex;
        private JobStealthGameEnv _env;
        private GA _gaModel;

        private List<float> _rewardsMeanOverTime;
        private WindowGraph _graphReward;

        private float[,] _currentSates;

        private void Awake()
        {
            _env = FindObjectOfType<JobStealthGameEnv>();

            _rewardsMeanOverTime = new List<float>(numberOfEpisodes);
            for (int i = 0; i < _rewardsMeanOverTime.Capacity; i++)
            {
                _rewardsMeanOverTime.Add(0f);
            }
        }

        private void Start()
        {
            //Temp seed
            Random.InitState(42);

            _env.CreatePopulation(populationSize);
            _currentSates = _env.DistributedResetEnv();

            var network = new NetworkLayer[]
            {
                new GANetworkLayer(populationSize, mutationNoiseStd, _env.GetObservationSize, 128,
                    ActivationFunction.Tanh, Instantiate(shader)),
                new GANetworkLayer(populationSize, mutationNoiseStd, 128, 128, ActivationFunction.Tanh,
                    Instantiate(shader)),
                new GANetworkLayer(populationSize, mutationNoiseStd, 128, _env.GetNumberOfActions,
                    ActivationFunction.Linear, Instantiate(shader))
            };

            var neModel = new GAModel(network);
            _gaModel = new GA(neModel, _env.GetNumberOfActions, populationSize, eliteNumber, tournamentNumber);

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

            var actions = _gaModel.SamplePopulationActions(_currentSates);
            var stepInfo = _env.DistributedStep(actions);

            _gaModel.AddExperience(stepInfo.Rewards, stepInfo.Dones);
            _currentSates = stepInfo.Observations;

            if (_gaModel.FinishedIndividuals < populationSize) return;

            //_gaModel.DoNoveltySearch(_env.GetPlayersPositions());
            _gaModel.Train();

            _rewardsMeanOverTime[_episodeIndex] = _gaModel.EpisodeRewardMean;

            _currentSates = _env.DistributedResetEnv();
            ++_episodeIndex;
        }

        private void PlotTrainingData()
        {
            Time.timeScale = 1;

            float rewardSum = 0.0f;
            foreach (var reward in _rewardsMeanOverTime)
            {
                rewardSum += reward;
            }

            // float timeSum = 0.0f;
            // foreach (var time in _times)
            // {
            //     timeSum += time;
            // }

            print("Average Reward: " + rewardSum / _rewardsMeanOverTime.Count);
            //print("Average Time: " + timeSum / _times.Count);

            var layoutGroup = FindObjectOfType<VerticalLayoutGroup>();
            _graphReward = Instantiate(windowGraphPrefab, layoutGroup.transform);
            StartCoroutine(ShowGraphs());
        }

        private IEnumerator ShowGraphs()
        {
            yield return new WaitForSeconds(0.1f);

            _graphReward.SetGraph(null, _rewardsMeanOverTime, GraphType.LineGraph,
                "Rewards per Episode", "episodes", "rewards");
        }

        private void OnDestroy()
        {
            Time.timeScale = 1;
            _gaModel?.Dispose();
            if (_env) _env.Close();
        }
    }
}