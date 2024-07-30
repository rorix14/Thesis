using System;
using System.Collections;
using System.Collections.Generic;
using Algorithms.RL;
using DL;
using DL.NN;
using Graphs;
using Gym;
using TestGround.Base;
using UnityEngine;
using UnityEngine.UI;
using Random = UnityEngine.Random;

namespace TestGround
{
    public class TestDQN : TestAlgorithmBase
    {
        [SerializeField] protected ComputeShader shader;
        [SerializeField] private WindowGraph windowGraphPrefab;
        [SerializeField] protected float simulationSpeed;
        [SerializeField] private int numberOfEpisodes;
        [SerializeField] protected int skippedFrames;
        [SerializeField] protected int targetNetworkCopyPeriod;
        private int _episodeIndex;
        protected float[] _currentSate;
        protected float[] _nextSate;
        private int _totalIteration;
        protected StealthGameEnv _env;

        // Cashed variables
        private int _currentSkippedFrame;
        private int _action;
        private float _reward;
        protected int _envStateSize;

        // DQN specific variables
        protected ModelDQN _DQN;
        protected float _epsilon;

        private WindowGraph _graphReward;
        private WindowGraph _graphLoss;

        // private Stopwatch _stopwatch;
        // private List<long> _times;

        [SerializeField] protected int batchSize;
        [SerializeField] protected float gamma;

        //Only here for testing
        [SerializeField] protected ActivationFunction activationFunction;
        [SerializeField] protected float learningRate;
        [SerializeField] protected float decayRate;
        [SerializeField] protected int neuronNumber;
        [SerializeField] protected float weightsInitStd;

        // public List<int> actions;
        // public int[] episodesLength;
        // public List<int> actionsPrev;
        // public int[] episodesLengthPrev;

        public override string GetDescription()
        {
            return "DQN, 3 layers, " + neuronNumber + " neurons, " + activationFunction +
                   ", " + batchSize + " batch size, " + gamma + " gamma, " + targetNetworkCopyPeriod +
                   "  copy network, lr " + learningRate + ", decay " + decayRate + ", initialization std " +
                   weightsInitStd;
        }

        private void Awake()
        {
            // foreach (var env in FindObjectsOfType<StealthGameEnv>())
            // {
            //     if (env.GetType() != typeof(StealthGameEnv)) continue;
            //
            //     _env = env;
            //     break;
            // }

            _env = FindObjectOfType<ImageStealthGameEnv>();
            Rewards = new List<float>(numberOfEpisodes);
            Loss = new List<float>(numberOfEpisodes);
            for (int i = 0; i < Rewards.Capacity; i++)
            {
                Rewards.Add(0f);
                Loss.Add(0f);
            }

            skippedFrames = skippedFrames > 0 ? skippedFrames : 1;

            // _stopwatch = new Stopwatch(); 
            // _times = new List<long>(1000000);
            // actions = new List<int>();
            // episodesLength = new int[numberOfEpisodes];
            // Random.InitState(42);
        }

        protected virtual void Start()
        {
            _currentSate = _env.ResetEnv();
            // var resetSate = _env.ResetEnv();
            // _envStateSize = _env.GetObservationSize * skippedFrames;
            // _nextSate = new float[_envStateSize];
            // _currentSate = new float[_envStateSize];
            // var startSateIndex = _envStateSize - _env.GetObservationSize;
            // for (int i = 0; i < _env.GetObservationSize; i++)
            // {
            //     _currentSate[startSateIndex + i] = resetSate[i];
            // }

            var updateLayers = new Layer[]
            {
                new NetworkLayer(_env.GetObservationSize * skippedFrames, neuronNumber, activationFunction,
                    Instantiate(shader), true, paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader),
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader),
                    paramsCoefficient: weightsInitStd)
            };
            var updateModel = new NetworkModel(updateLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            var targetLayers = new Layer[]
            {
                new NetworkLayer(_env.GetObservationSize * skippedFrames, neuronNumber, activationFunction,
                    Instantiate(shader), true, paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader),
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader),
                    paramsCoefficient: weightsInitStd)
            };
            var targetModel = new NetworkModel(targetLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            _DQN = new ModelDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize,
                batchSize: batchSize, gamma: gamma);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }

        private void FixedUpdate()
        {
            if (_episodeIndex >= numberOfEpisodes)
            {
                IsFinished = true;

                // if (!_env) return;
                // _env.Close();
                // PlotTrainingData();
                return;
            }

            //_stopwatch.Restart();

            _action = _currentSkippedFrame == 0 ? _DQN.EpsilonGreedySample(_currentSate, _epsilon) : _action;
            ++_currentSkippedFrame;

            var skippFrame = _currentSkippedFrame < skippedFrames;
            var stepInfo = _env.Step(_action, skippFrame);

            _reward += stepInfo.Reward;

            _nextSate = stepInfo.Observation;
            // var nextStateStartIndex = _currentSkippedFrame * _env.GetObservationSize;
            // for (int i = 0; i < _env.GetObservationSize; i++)
            // {
            //     _nextSate[nextStateStartIndex + i] = stepInfo.Observation[i];
            // }

            // _currentSkippedFrame = (_currentSkippedFrame + 1) % skippedFrames;
            if (!stepInfo.Done && skippFrame) return;

            _DQN.AddExperience(_currentSate, _action, _reward, stepInfo.Done, _nextSate);

            _DQN.Train();

            if (_totalIteration % targetNetworkCopyPeriod == 0)
            {
                _DQN.SetTargetModel();
            }

            Rewards[_episodeIndex] += _reward;

            _currentSate = _nextSate;
            // _nextSate = new float[_envStateSize];
            _reward = 0f;
            _currentSkippedFrame = 0;
            ++_totalIteration;

            // actions.Add(action);
            // episodesLength[_episodeIndex] += 1;

            if (!stepInfo.Done)
            {
                // _stopwatch.Stop();
                // _times.Add(_stopwatch.ElapsedMilliseconds);
                return;
            }

            Loss[_episodeIndex] = _DQN.SampleLoss();

            _currentSate = _env.ResetEnv();
            // var resetSate = _env.ResetEnv();
            // _currentSate = new float[_envStateSize];
            // var startSateIndex = _envStateSize - _env.GetObservationSize;
            // for (int i = 0; i < _env.GetObservationSize; i++)
            // {
            //     _currentSate[startSateIndex + i] = resetSate[i];
            // }

            // _currentSkippedFrame = 0;
            ++_episodeIndex;

            // could clamp epsilon value with a max function in order to have a min exploration value
            _epsilon = 1.0f / (float)Math.Sqrt(_episodeIndex + 1);
            //_epsilon = 0.0f;

            // _stopwatch.Stop();
            // _times.Add(_stopwatch.ElapsedMilliseconds);
        }

        private void PlotTrainingData()
        {
            Time.timeScale = 1;

            float rewardSum = 0.0f;
            foreach (var reward in Rewards)
            {
                rewardSum += reward;
            }

            float lossSum = 0.0f;
            for (int i = 0; i < Loss.Count; i++)
            {
                var loss = Loss[i];
                lossSum += loss;
                // if (loss > 1)
                // {
                //     _lossPerEpisode[i] = 1.0f;
                // }
            }

            // float timeSum = 0.0f;
            // foreach (var time in _times)
            // {
            //     timeSum += time;
            // }

            print("Average Reward: " + rewardSum / Rewards.Count);
            print("Average Loss: " + lossSum / Loss.Count);
            //print("Average Time: " + timeSum / _times.Count);

            var layoutGroup = FindObjectOfType<VerticalLayoutGroup>();
            _graphReward = Instantiate(windowGraphPrefab, layoutGroup.transform);
            _graphLoss = Instantiate(windowGraphPrefab, layoutGroup.transform);
            StartCoroutine(ShowGraphs());

            // EditorApplication.Beep();
            // EditorApplication.ExitPlaymode();
        }

        private IEnumerator ShowGraphs()
        {
            //testing images env
            // for (int i = 0; i < numberOfEpisodes; i++)
            // {
            //     if (episodesLength[i] != episodesLengthPrev[i])
            //     {
            //         print("Episode length not the same at " + i + ". New: " + episodesLength[i] + ", old: " +
            //               episodesLengthPrev[i]);
            //     }
            // }
            //
            // for (int i = 0; i < actions.Count && i < actionsPrev.Count; i++)
            // {
            //     if (actions[i] != actionsPrev[i])
            //     {
            //         print("Actions not the same at " + i + ". New: " + actions[i] + ", old: " +
            //               actionsPrev[i]);
            //     }
            // }

            yield return new WaitForSeconds(0.1f);

            _graphReward.SetGraph(null, Rewards, GraphType.LineGraph,
                "Rewards per Episode", "episodes", "rewards");

            _graphLoss.SetGraph(null, Loss, GraphType.LineGraph,
                "Loss per Episode", "episodes", "loss");
        }

        private void OnDestroy()
        {
            //Time.timeScale = 1;
            _DQN?.Dispose();
        }
    }
}