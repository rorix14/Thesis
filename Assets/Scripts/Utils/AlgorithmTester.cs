using System;
using System.Diagnostics;
using TestGround.Base;
using UnityEditor;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Utils
{
    public class AlgorithmTester : MonoBehaviour
    {
        [SerializeField] protected TestAlgorithmBase[] algorithmsPrefabs;
        protected string[] _testDescriptions;
        [SerializeField] protected int testNumber;
        [SerializeField] private int generatorSeed = 42;
        [SerializeField] protected bool saveInfo;
        [SerializeField] private string saveFileName;

        protected TestAlgorithmBase _currentAlgorithm;
        protected int _currentAlgorithmIndex;
        protected int _currenTest;
        protected int[] _testSeeds;
        protected bool _testsFinished;

        private TestResultsSavable[] _testResults;

        protected float _rewardAverageFinal;
        protected float _lossAverageFinal;

        private Stopwatch _stopwatch;

        protected virtual void Awake()
        {
            Random.InitState(generatorSeed);

            _testSeeds = new int [testNumber];

            var iteration = 0;
            while (iteration < testNumber)
            {
                var hasSeed = false;
                var randomSeed = Random.Range(1, 100);

                for (int i = 0; i < iteration + 1; i++)
                {
                    if (_testSeeds[i] != randomSeed) continue;

                    hasSeed = true;
                    break;
                }

                if (hasSeed) continue;

                _testSeeds[iteration] = randomSeed;
                ++iteration;
            }

            _testResults = new TestResultsSavable[algorithmsPrefabs.Length];
            _testDescriptions = new string[algorithmsPrefabs.Length];
            for (int i = 0; i < algorithmsPrefabs.Length; i++)
            {
                _testResults[i].algorithmStatsArray = new AlgorithmStats[testNumber];
                _testDescriptions[i] = algorithmsPrefabs[i].GetDescription();
            }

            _stopwatch = new Stopwatch();
        }

        protected virtual void Start()
        {
            if (algorithmsPrefabs.Length == 0) return;

            Random.InitState(_testSeeds[_currenTest]);
            //Random.InitState(36);
            _currentAlgorithm = Instantiate(algorithmsPrefabs[_currentAlgorithmIndex]);

            _stopwatch.Start();
        }

        protected virtual void FixedUpdate()
        {
            if (_testsFinished || !_currentAlgorithm.IsFinished) return;

            _stopwatch.Stop();

            PrintAlgorithmResults();
            StoreData();
            Destroy(_currentAlgorithm.gameObject);

            _currenTest++;
            if (_currenTest >= testNumber)
            {
                print(_testDescriptions[_currentAlgorithmIndex] + ". Final: reward average: " +
                      _rewardAverageFinal / testNumber + ", loss average: " + _lossAverageFinal / testNumber);

                _rewardAverageFinal = 0f;
                _lossAverageFinal = 0f;

                _currenTest = 0;
                _currentAlgorithmIndex++;

                if (_currentAlgorithmIndex >= algorithmsPrefabs.Length)
                {
                    if (saveInfo)
                    {
                        SaveTestDataToFile();
                    }

                    _testsFinished = true;

                    EditorApplication.Beep();
                    EditorApplication.ExitPlaymode();
                    return;
                }
            }

            Random.InitState(_testSeeds[_currenTest]);
            _currentAlgorithm = Instantiate(algorithmsPrefabs[_currentAlgorithmIndex]);

            _stopwatch.Restart();
        }

        private void PrintAlgorithmResults()
        {
            var rewardMean = 0f;
            var lossMean = 0f;

            var size = _currentAlgorithm.Rewards.Count;
            for (int i = 0; i < size; i++)
            {
                rewardMean += _currentAlgorithm.Rewards[i];
                lossMean += _currentAlgorithm.Loss[i];
            }

            rewardMean /= size;
            lossMean /= size;

            print("Reward mean: " + rewardMean + ", loss mean: " + lossMean);

            _rewardAverageFinal += rewardMean;
            _lossAverageFinal += lossMean;
        }

        private void StoreData()
        {
            _testResults[_currentAlgorithmIndex].averageTrainTime = _stopwatch.ElapsedMilliseconds;
            _testResults[_currentAlgorithmIndex].algorithmStatsArray[_currenTest].rewards =
                _currentAlgorithm.Rewards.ToArray();
            _testResults[_currentAlgorithmIndex].algorithmStatsArray[_currenTest].loss =
                _currentAlgorithm.Loss.ToArray();
        }

        private void SaveTestDataToFile()
        {
            for (int i = 0; i < algorithmsPrefabs.Length; i++)
            {
                _testResults[i].description = _testDescriptions[i];
                _testResults[i].averageTrainTime /= testNumber;
            }

            FileHandler.SaveToJson(_testResults, "Test Data/Preliminary algorithm Tests/" + saveFileName);
        }

        [Serializable]
        private struct TestResultsSavable
        {
            public string description;
            public long averageTrainTime;
            public AlgorithmStats[] algorithmStatsArray;

            public TestResultsSavable(string description, long averageTrainTime, AlgorithmStats[] algorithmStatsArray)
            {
                this.description = description;
                this.averageTrainTime = averageTrainTime;
                this.algorithmStatsArray = algorithmStatsArray;
            }
        }

        [Serializable]
        private struct AlgorithmStats
        {
            public float[] rewards;
            public float[] loss;

            public AlgorithmStats(float[] rewards, float[] loss)
            {
                this.rewards = rewards;
                this.loss = loss;
            }
        }
    }
}