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
        [SerializeField] private string graph1Title;
        [SerializeField] private string graph1LabelX;
        [SerializeField] private string graph1LabelY;
        [SerializeField] private string graph1ImageName;
        [SerializeField] private string graph2Title;
        [SerializeField] private string graph2LabelX;
        [SerializeField] private string graph2LabelY;
        [SerializeField] private string graph2ImageName;

        private string[] _algorithmsNames;
        private List<float>[] _testsRewards;
        private List<float>[] _testsLosses;

        //Cashed variables
        private float[][] _movingAverages;

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
            var graphLoss = Instantiate(windowGraphPrefab, layoutGroupTransform);

            StartCoroutine(DisplayGraphs(graphReward, graphLoss));
        }

        private IEnumerator DisplayGraphs(WindowGraph graphReward, WindowGraph graphLoss)
        {
            yield return Enumerator(graphReward, graphLoss);
            CalculateMovingAverage(3);
            yield return Enumerator(graphReward, graphLoss, " Smooth");
        }

        private IEnumerator Enumerator(WindowGraph graphReward, WindowGraph graphLoss, string prefix = "")
        {
            graphLoss.gameObject.SetActive(false);
            graphReward.gameObject.SetActive(true);
            yield return new WaitForSeconds(0.1f);
            graphReward.SetStaticGraph(_testsRewards, graph1Title, graph1LabelX, graph1LabelY, _algorithmsNames);

            yield return new WaitForEndOfFrame();
            ScreenCapture.CaptureScreenshot(string.Concat(graph1ImageName, prefix, ".png"), 4);
            yield return new WaitForSeconds(0.1f);

            graphReward.gameObject.SetActive(false);
            graphLoss.gameObject.SetActive(true);
            yield return new WaitForSeconds(0.1f);
            graphLoss.SetStaticGraph(_testsLosses, graph2Title, graph2LabelX, graph2LabelY, _algorithmsNames);

            yield return new WaitForEndOfFrame();
            ScreenCapture.CaptureScreenshot(string.Concat(graph2ImageName, prefix, ".png"), 4);
            yield return new WaitForSeconds(0.1f);

            graphReward.gameObject.SetActive(true);
            yield return new WaitForSeconds(0.1f);
            graphReward.SetStaticGraph(_testsRewards, graph1Title, graph1LabelX, graph1LabelY, _algorithmsNames);
            graphLoss.SetStaticGraph(_testsLosses, graph2Title, graph2LabelX, graph2LabelY, _algorithmsNames);

            yield return new WaitForEndOfFrame();
            ScreenCapture.CaptureScreenshot(string.Concat(graph1ImageName, graph2ImageName, prefix, ".png"), 4);
            yield return new WaitForSeconds(0.1f);

            for (int i = 0; i < _testsRewards.Length; i++)
            {
                var algorithmName = _algorithmsNames[i];

                graphLoss.gameObject.SetActive(false);
                graphReward.gameObject.SetActive(true);
                yield return new WaitForSeconds(0.1f);
                var title = string.Concat(algorithmName, " ", graph1Title);
                graphReward.SetStaticGraph(_testsRewards, graph1Title, graph1LabelX, graph1LabelY, _algorithmsNames);

                graphReward.SetGraph(new List<float>(), _testsRewards[i], GraphType.LineGraph, title, graph1LabelX,
                    graph1LabelY);

                yield return new WaitForEndOfFrame();
                ScreenCapture.CaptureScreenshot(string.Concat(algorithmName, " ", graph1ImageName, prefix, ".png"), 4);
                yield return new WaitForSeconds(0.1f);

                graphReward.gameObject.SetActive(false);
                graphLoss.gameObject.SetActive(true);
                yield return new WaitForSeconds(0.1f);
                title = string.Concat(algorithmName, " ", graph2Title);
                graphLoss.SetGraph(new List<float>(), _testsLosses[i], GraphType.LineGraph, title, graph2LabelX,
                    graph2LabelY);

                yield return new WaitForEndOfFrame();
                ScreenCapture.CaptureScreenshot(string.Concat(algorithmName, " " + graph2ImageName, prefix, ".png"), 4);
                yield return new WaitForSeconds(0.1f);
            }
        }

        private void CalculateMovingAverage(int windowSize)
        {
            var movingAveragesReward = new float[_testsRewards[0].Count - windowSize + 1];
            var movingAveragesLoss = new float[_testsRewards[0].Count - windowSize + 1];

            for (int k = 0; k < _testsRewards.Length; k++)
            {
                var data1 = _testsRewards[k];
                var data2 = _testsLosses[k];

                for (int i = 0; i < movingAveragesReward.Length; i++)
                {
                    var sum1 = 0f;
                    var sum2 = 0f;
                    for (int j = i; j < i + windowSize; j++)
                    {
                        sum1 += data1[j];
                        sum2 += data2[j];
                    }

                    movingAveragesReward[i] = sum1 / windowSize;
                    movingAveragesLoss[i] = sum2 / windowSize;
                }

                for (int i = 0; i < movingAveragesReward.Length; i++)
                {
                    _testsRewards[k][i] = movingAveragesReward[i];
                    _testsLosses[k][i] = movingAveragesLoss[i];
                }
            }
        }
    }
}