using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace Graphs
{
    public enum GraphType
    {
        LineGraph,
        BarGraph
    }

    public class WindowGraph : MonoBehaviour
    {
        [SerializeField] private TextMeshProUGUI graphTitle;
        [SerializeField] private TextMeshProUGUI xGraphLabel;
        [SerializeField] private TextMeshProUGUI yGraphLabel;
        [SerializeField] private RectTransform graphContainer;
        [SerializeField] private RectTransform labelTemplateX;
        [SerializeField] private RectTransform labelTemplateY;
        [SerializeField] private RectTransform dashTemplateX;
        [SerializeField] private RectTransform dashTemplateY;
        private List<RectTransform> _xLabelList;
        private List<RectTransform> _yLabelList;
        private List<RectTransform> _xDashList;
        private List<RectTransform> _yDashList;

        private List<float> _xValueList;
        private List<float> _yValueList;

        [SerializeField] private int xMaxSeparatorCount = 15;
        [SerializeField] private int yMaxSeparatorCount = 15;

        private GraphVisual _graphVisual;

        private void Awake()
        {
            _xLabelList = new List<RectTransform>();
            _yLabelList = new List<RectTransform>();
            _xDashList = new List<RectTransform>();
            _yDashList = new List<RectTransform>();

            _graphVisual = null;

            _xValueList = new List<float>();
            _yValueList = new List<float>();
        }
        
        public void SetGraph(List<float> xValueList, List<float> yValueList, GraphType graphType, string title,
            string xLabel, string yLabel)
        {
            _xValueList = xValueList;
            _yValueList = yValueList;

            _graphVisual = graphType switch
            {
                GraphType.LineGraph => new LineGraphVisual(graphContainer, Color.white, 5f, 0f),
                GraphType.BarGraph => new BarChartVisual(graphContainer, Color.yellow, 0.9f, 0.5f),
                _ => _graphVisual
            };

            graphTitle.text = title;
            xGraphLabel.text = string.Concat("x= ", xLabel);
            yGraphLabel.text = string.Concat("y= ", yLabel);
            InitializeGraph(_yValueList, _graphVisual);
        }

        private void InitializeGraph(IReadOnlyList<float> valueList, GraphVisual graphVisual)
        {
            foreach (var xLabel in _xLabelList)
                Destroy(xLabel);

            foreach (var yLabel in _yLabelList)
                Destroy(yLabel);

            foreach (var xDash in _xDashList)
                Destroy(xDash);

            foreach (var yDash in _yDashList)
                Destroy(yDash);

            _xLabelList.Clear();
            _yLabelList.Clear();
            _xDashList.Clear();
            _yDashList.Clear();

            graphVisual.CleanUp();

            int valueListSize = valueList.Count;
            var sizeDelta = graphContainer.sizeDelta;
            float graphWidth = sizeDelta.x;
            float graphHeight = sizeDelta.y;
            float xSize = graphWidth / valueListSize;
            var xPadding = xSize * graphVisual.PaddingPercentage;

            bool valueIsLess = valueListSize <= xMaxSeparatorCount;
            int separatorCount = valueIsLess ? valueListSize : xMaxSeparatorCount;

            for (int i = 0; i < separatorCount; i++)
            {
                float normalizedValue = i / (float)separatorCount;
                float xPos = valueIsLess ? xPadding + i * xSize : xPadding + normalizedValue * graphWidth;

                // TODO: quick fix for the x label text so it represents a custom array of values,
                // should do the same for the other functions that set the x label 
                //string labelText = ((i + 1) / (float)separatorCount * _xValueList[_xValueList.Count - 1]).ToString("F1");

                var dashX = Instantiate(dashTemplateX, graphContainer, false);
                dashX.gameObject.SetActive(true);
                dashX.anchoredPosition = new Vector2(xPos, 0);
                _xDashList.Add(dashX);

                var labelX = Instantiate(labelTemplateX, graphContainer, false);
                labelX.gameObject.SetActive(true);
                labelX.anchoredPosition = new Vector2(xPos, -25f);
                labelX.GetComponent<TextMeshProUGUI>().text = (normalizedValue * valueListSize).ToString("F1");
                _xLabelList.Add(labelX);
            }

            CalculateYScale(out var yMin, out var yMax);

            for (int i = 0; i <= yMaxSeparatorCount; i++)
            {
                float normalizedValue = i / (float)yMaxSeparatorCount;

                var dashY = Instantiate(dashTemplateY, graphContainer, false);
                dashY.gameObject.SetActive(true);
                dashY.anchoredPosition = new Vector2(0, normalizedValue * graphHeight);
                _yDashList.Add(dashY);

                var labelY = Instantiate(labelTemplateY, graphContainer, false);
                labelY.gameObject.SetActive(true);
                labelY.anchoredPosition = new Vector2(-30f, normalizedValue * graphHeight);
                labelY.GetComponent<TextMeshProUGUI>().text = (yMin + normalizedValue * (yMax - yMin)).ToString("F1");
                _yLabelList.Add(labelY);
            }

            graphVisual.CreateGraphVisual(valueList, yMin, yMax);
        }

        private void AddValue(GraphVisual graphVisual, int value)
        {
            CalculateYScale(out var yMinBefore, out var yMaxBefore);
            _yValueList.Add(value);
            CalculateYScale(out var yMin, out var yMax);

            bool yScaleChanged = Mathf.Abs(yMinBefore - yMin) > 0 || Mathf.Abs(yMaxBefore - yMax) > 0;
            float xSize = graphContainer.sizeDelta.x / _yValueList.Count;
            var xPadding = xSize * graphVisual.PaddingPercentage;

            bool xScaleChanged = _yValueList.Count <= xMaxSeparatorCount;
            if (xScaleChanged)
            {
                var dashX = Instantiate(dashTemplateX, graphContainer, false);
                dashX.gameObject.SetActive(true);
                _xDashList.Add(dashX);

                var labelX = Instantiate(labelTemplateX, graphContainer, false);
                labelX.gameObject.SetActive(true);
                _xLabelList.Add(labelX);
            }

            graphVisual.UpdateAllGraphElements(_yValueList, yMin, yMax);

            for (int i = 0; i < _xLabelList.Count; i++)
            {
                float normalizedValue = i / (float)_xLabelList.Count;
                _xLabelList[i].GetComponent<TextMeshProUGUI>().text =
                    (normalizedValue * _yValueList.Count).ToString("F1");

                if (!xScaleChanged) continue;

                float xPos = xPadding + i * xSize;
                _xDashList[i].anchoredPosition = new Vector2(xPos, 0);
                _xLabelList[i].anchoredPosition = new Vector2(xPos, -25f);
            }

            if (!yScaleChanged) return;
            for (int i = 0; i < _yLabelList.Count; i++)
            {
                float normalizedValue = i / (float)_yLabelList.Count;
                _yLabelList[i].GetComponent<TextMeshProUGUI>().text =
                    (yMin + normalizedValue * (yMax - yMin)).ToString("F1");
            }
        }

        private void UpdateValue(GraphVisual graphVisual, int index, int value)
        {
            CalculateYScale(out var yMinBefore, out var yMaxBefore);
            _yValueList[index] = value;
            CalculateYScale(out var yMin, out var yMax);

            bool yScaleChanged = Mathf.Abs(yMinBefore - yMin) > 0 || Mathf.Abs(yMaxBefore - yMax) > 0;
            if (!yScaleChanged)
            {
                graphVisual.UpdateGraphElement(index, value, yMin, yMax);
                return;
            }

            graphVisual.UpdateAllGraphElements(_yValueList, yMin, yMax);

            for (int i = 0; i < _yLabelList.Count; i++)
            {
                float normalizedValue = i / (float)_yLabelList.Count;
                _yLabelList[i].GetComponent<TextMeshProUGUI>().text =
                    (yMin + normalizedValue * (yMax - yMin)).ToString("F1");
            }
        }

        private void CalculateYScale(out float yMin, out float yMax)
        {
            yMin = _yValueList[0];
            yMax = _yValueList[0];

            foreach (var value in _yValueList)
            {
                if (value > yMax)
                    yMax = value;

                if (value < yMin)
                    yMin = value;
            }

            bool startYScaleAtZero = yMin >= 0.0f;

            var buffer = (yMax - yMin) * 0.1f;
            yMax += buffer;
            yMin -= buffer;

            if (startYScaleAtZero)
                yMin = 0;
        }

        private abstract class GraphVisual
        {
            public float PaddingPercentage;
            protected RectTransform GraphContainer;
            protected Color GraphColor;

            public abstract void CreateGraphVisual(IReadOnlyList<float> values, float yMin, float yMax);
            public abstract void UpdateGraphElement(int index, float value, float yMin, float yMax);
            public abstract void UpdateAllGraphElements(IReadOnlyList<float> values, float yMin, float yMax);
            public abstract void CleanUp();
        }

        private class BarChartVisual : GraphVisual
        {
            private readonly List<RectTransform> _barList;
            private readonly float _barWidthMult;

            public BarChartVisual(RectTransform graphContainer, Color barColor, float barWidthMult,
                float paddingPercentage)
            {
                GraphContainer = graphContainer;
                GraphColor = barColor;
                PaddingPercentage = paddingPercentage;
                _barWidthMult = barWidthMult;
                _barList = new List<RectTransform>();
            }

            public override void CreateGraphVisual(IReadOnlyList<float> values, float yMin, float yMax)
            {
                var containerSize = GraphContainer.sizeDelta;
                var xSize = containerSize.x / values.Count;
                var xPadding = xSize * PaddingPercentage;

                for (int i = 0; i < values.Count; i++)
                {
                    float xPos = xPadding + i * xSize;
                    float yPos = (values[i] - yMin) / (yMax - yMin) * containerSize.y;
                    _barList.Add(CreateBar(new Vector2(xPos, yPos), xSize));
                }
            }

            public override void UpdateGraphElement(int index, float value, float yMin, float yMax)
            {
                var containerSize = GraphContainer.sizeDelta;
                float yPos = (value - yMin) / (yMax - yMin) * containerSize.y;

                var bar = _barList[index];
                bar.sizeDelta = new Vector2(bar.sizeDelta.x, yPos);
            }

            public override void UpdateAllGraphElements(IReadOnlyList<float> values, float yMin, float yMax)
            {
                var containerSize = GraphContainer.sizeDelta;
                var xSize = containerSize.x / values.Count;

                for (int i = 0; i < values.Count; i++)
                {
                    float xPos = xSize * PaddingPercentage + i * xSize;
                    float yPos = (values[i] - yMin) / (yMax - yMin) * containerSize.y;

                    if (i >= _barList.Count)
                    {
                        _barList.Add(CreateBar(new Vector2(xPos, yPos), xSize));
                    }
                    else
                    {
                        var bar = _barList[i];
                        bar.anchoredPosition = new Vector2(xPos, 0f);
                        bar.sizeDelta = new Vector2(xSize * _barWidthMult, yPos);
                    }
                }
            }

            private RectTransform CreateBar(Vector2 graphPos, float barWidth)
            {
                var bar = new GameObject("bar", typeof(Image));
                bar.transform.SetParent(GraphContainer, false);
                bar.GetComponent<Image>().color = GraphColor;

                var rectTransform = bar.GetComponent<RectTransform>();
                rectTransform.anchoredPosition = new Vector2(graphPos.x, 0f);
                rectTransform.sizeDelta = new Vector2(barWidth * _barWidthMult, graphPos.y);
                rectTransform.anchorMin = new Vector2(0, 0);
                rectTransform.anchorMax = new Vector2(0, 0);
                rectTransform.pivot = new Vector2(0.5f, 0f);
                return rectTransform;
            }

            public override void CleanUp()
            {
                foreach (var bar in _barList)
                    Destroy(bar);

                _barList.Clear();
            }
        }

        private class LineGraphVisual : GraphVisual
        {
            private readonly List<RectTransform> _lineList;
            private readonly float _lineThickness;

            public LineGraphVisual(RectTransform graphContainer, Color lineColor, float lineThickness,
                float paddingPercentage)
            {
                GraphContainer = graphContainer;
                GraphColor = lineColor;
                PaddingPercentage = paddingPercentage;
                _lineThickness = lineThickness;
                _lineList = new List<RectTransform>();
            }

            public override void CreateGraphVisual(IReadOnlyList<float> values, float yMin, float yMax)
            {
                var containerSize = GraphContainer.sizeDelta;
                var xSize = containerSize.x / values.Count;
                var xPadding = xSize * PaddingPercentage;

                var line = CreateLine();
                var lineRenderer = line.GetComponent<LineRendererHUD>();
                lineRenderer.points = new List<Vector2>(values.Count);

                for (int i = 0; i < values.Count; i++)
                {
                    float xPos = xPadding + i * xSize;
                    float yPos = (values[i] - yMin) / (yMax - yMin) * containerSize.y;
                    lineRenderer.points.Add(new Vector2(xPos, yPos));
                }

                _lineList.Add(line);
                lineRenderer.SetVerticesDirty();
            }

            public override void UpdateGraphElement(int index, float value, float yMin, float yMax)
            {
                float yPos = (value - yMin) / (yMax - yMin) * GraphContainer.sizeDelta.y;

                var lineRenderer = _lineList[0].GetComponent<LineRendererHUD>();
                lineRenderer.points[index] = new Vector2(lineRenderer.points[index].x, yPos);
                lineRenderer.SetVerticesDirty();
            }

            public override void UpdateAllGraphElements(IReadOnlyList<float> values, float yMin, float yMax)
            {
                var containerSize = GraphContainer.sizeDelta;
                var xSize = containerSize.x / values.Count;

                var lineRenderer = _lineList[0].GetComponent<LineRendererHUD>();
                for (int i = 0; i < values.Count; i++)
                {
                    float xPos = xSize * PaddingPercentage + i * xSize;
                    float yPos = (values[i] - yMin) / (yMax - yMin) * containerSize.y;

                    if (i >= lineRenderer.points.Count)
                    {
                        lineRenderer.points.Add(new Vector2(xPos, yPos));
                    }
                    else
                    {
                        lineRenderer.points[i] = new Vector2(xPos, yPos);
                    }
                }

                lineRenderer.SetVerticesDirty();
            }

            private RectTransform CreateLine()
            {
                var line = new GameObject("line", typeof(CanvasRenderer), typeof(LineRendererHUD));
                line.transform.SetParent(GraphContainer, false);
                var lineRenderer = line.GetComponent<LineRendererHUD>();
                lineRenderer.color = GraphColor;
                lineRenderer.thickness = _lineThickness;

                var rectTransform = line.GetComponent<RectTransform>();
                rectTransform.anchoredPosition = new Vector2(0f, 0f);
                rectTransform.anchorMin = new Vector2(0, 0);
                rectTransform.anchorMax = new Vector2(1, 1);
                rectTransform.pivot = new Vector2(0f, 0f);
                return rectTransform;
            }

            public override void CleanUp()
            {
                foreach (var line in _lineList)
                    Destroy(line);

                _lineList.Clear();
            }
        }
    }
}