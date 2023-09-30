using System;
using System.Collections;
using System.Collections.Generic;
using Graphs;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using Random = UnityEngine.Random;

namespace Utils
{
    public class AlgorithmPlotter : AlgorithmTester
    {
        [SerializeField] private int[] testSeeds;
        [SerializeField] private WindowGraph windowGraphPrefab;

        private string[] _algorithmsNames;
        private List<float>[] _testsRewards;
        private List<float>[] _testsLosses;

        protected override void Awake()
        {
            testNumber = 1;
            saveInfo = false;

            _testSeeds = new int[testSeeds.Length];
            _testsRewards = new List<float>[testSeeds.Length];
            _testsLosses = new List<float>[testSeeds.Length];
            for (int i = 0; i < testSeeds.Length; i++)
            {
                _testSeeds[i] = testSeeds[i];
                _testsRewards[i] = new List<float>();
                _testsLosses[i] = new List<float>();
            }

            _algorithmsNames = new string[algorithmsPrefabs.Length];
            _testDescriptions = new string[algorithmsPrefabs.Length];
            for (int i = 0; i < algorithmsPrefabs.Length; i++)
            {
                var description = algorithmsPrefabs[i].GetDescription();
                _testDescriptions[i] = description;

                var parts = description.Split(new char[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                _algorithmsNames[i] = parts[0];
            }
        }

        protected override void Start()
        {
            if (algorithmsPrefabs.Length != testSeeds.Length) return;

            Random.InitState(_testSeeds[0]);
            _currentAlgorithm = Instantiate(algorithmsPrefabs[0]);
        }

        protected override void FixedUpdate()
        {
            if (_testsFinished || !_currentAlgorithm.IsFinished) return;

            GatherData();
            Destroy(_currentAlgorithm.gameObject);

            print(_testDescriptions[_currentAlgorithmIndex] + ". Final: reward average: " +
                  _rewardAverageFinal + ", loss average: " + _lossAverageFinal);

            _rewardAverageFinal = 0f;
            _lossAverageFinal = 0f;

            _currentAlgorithmIndex++;

            if (_currentAlgorithmIndex >= algorithmsPrefabs.Length)
            {
                ShowGraphs();

                _testsFinished = true;

                EditorApplication.Beep();
                return;
            }

            Random.InitState(_testSeeds[_currentAlgorithmIndex]);
            _currentAlgorithm = Instantiate(algorithmsPrefabs[_currentAlgorithmIndex]);
        }

        private void GatherData()
        {
            var rewardMean = 0f;
            var lossMean = 0f;

            var size = _currentAlgorithm.Rewards.Count;
            var rewards = _testsRewards[_currentAlgorithmIndex];
            var losses = _testsLosses[_currentAlgorithmIndex];
            for (int i = 0; i < size; i++)
            {
                var reward = _currentAlgorithm.Rewards[i];
                var loss = _currentAlgorithm.Loss[i];
                rewards.Add(reward);
                losses.Add(loss);
                rewardMean += reward;
                lossMean += loss;
            }

            rewardMean /= size;
            lossMean /= size;

            _rewardAverageFinal += rewardMean;
            _lossAverageFinal += lossMean;
        }

        private void ShowGraphs()
        {
            Time.timeScale = 1;

            var layoutGroup = FindObjectOfType<VerticalLayoutGroup>();
            var layoutGroupTransform = layoutGroup.transform;
            layoutGroupTransform.parent.GetComponentInChildren<Image>().enabled = true;

            var graphReward = Instantiate(windowGraphPrefab, layoutGroupTransform);
            //var graphLoss = Instantiate(windowGraphPrefab, layoutGroupTransform);

            StartCoroutine(DisplayGraphs(graphReward, null));
        }

        private IEnumerator DisplayGraphs(WindowGraph graphReward, WindowGraph graphLoss)
        {
            yield return new WaitForSeconds(0.1f);

            //TODO: make strings into variables
            graphReward.SetStaticGraph(_testsRewards, "Rewards per Episode", "episodes", "rewards", _algorithmsNames);
            //graphLoss.SetStaticGraph(_testsLosses, "Loss per Episode", "episodes", "loss", _algorithmsNames);
        }
    }
}