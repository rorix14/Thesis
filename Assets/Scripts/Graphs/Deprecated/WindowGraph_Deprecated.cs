using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using Random = UnityEngine.Random;

namespace Graphs
{
    public class WindowGraph_Deprecated : MonoBehaviour
    {
        private static WindowGraph_Deprecated _instance;
        [SerializeField] private Sprite dotSprite;
        private RectTransform _graphContainer;
        private RectTransform _labelTemplateX;
        private RectTransform _labelTemplateY;
        private RectTransform _dashTemplateX;
        private RectTransform _dashTemplateY;
        private GameObject _tooltip;
        private List<GameObject> _gameObjects;
        private List<IGraphVisualObject> _graphVisualObjectList;
        private List<RectTransform> _yLabelList;

        private List<int> _valueList;
        private float _xSize;
        private bool _startYScaleAtZero;

        public List<Vector4> pointList;

        private BarChartVisual _barChartVisual;
        private LineGraphVisual _lineGraphVisual;

        private void Awake()
        {
            _instance = this;
            _startYScaleAtZero = true;
            _graphContainer = transform.Find("Graph Container").GetComponent<RectTransform>();
            _labelTemplateX = _graphContainer.Find("Label Template X").GetComponent<RectTransform>();
            _labelTemplateY = _graphContainer.Find("Label Template Y").GetComponent<RectTransform>();
            _dashTemplateY = _graphContainer.Find("Dash Template X").GetComponent<RectTransform>();
            _dashTemplateX = _graphContainer.Find("Dash Template Y").GetComponent<RectTransform>();
            _tooltip = _graphContainer.Find("Tooltip").gameObject;
            _gameObjects = new List<GameObject>();
            _graphVisualObjectList = new List<IGraphVisualObject>();
            _yLabelList = new List<RectTransform>();

            _valueList = new List<int>()
            {
                5, 98, 56, 45, /*30, 22, 17, 15, 13, 17, 25, 37, 40, 36, 33, 50, 8, 40,
                5, 98, 56, 45, 30, 22, 17, 15, 13, 17, 25, 37, 40, 36, 33, 50, 8, 40,*/
            };
            pointList = new List<Vector4>(_valueList.Count);

            //_barChartVisual = new BarChartVisual(_graphContainer, Color.yellow, 0.9f);
            //ShowGraph(_valueList, _barChartVisual);
            _lineGraphVisual = new LineGraphVisual(_graphContainer, dotSprite, Color.white, Color.red);
            ShowGraph(_valueList, _lineGraphVisual);
            
            var lineHud = FindObjectOfType<LineRendererHUD>();
            lineHud.points.Clear();
            foreach (var point in pointList)
            {
                lineHud.points.Add(point);
            }

            //StartCoroutine(RunEverySecond());
        }

        private IEnumerator RunEverySecond()
        {
            //int i = 0;
            while (true)
            {
                yield return new WaitForSeconds(1f);
                //UpdateValue(0, i++);
                pointList.Clear();
                _valueList.Add(Random.Range(0, 100));
                ShowGraph(_valueList, _lineGraphVisual);

                var lineHud = FindObjectOfType<LineRendererHUD>();
                lineHud.points.Clear();
                foreach (var point in pointList)
                {
                    lineHud.points.Add(point);
                }

                lineHud.SetVerticesDirty();
            }
        }

        private void ShowTooltip(string tooltipText, Vector2 anchoredPos)
        {
            _tooltip.SetActive(true);
            _tooltip.GetComponent<RectTransform>().anchoredPosition = anchoredPos;
            var tooltipTextComp = _tooltip.transform.Find("Text").GetComponent<TextMeshProUGUI>();
            tooltipTextComp.autoSizeTextContainer = true;
            tooltipTextComp.text = tooltipText;

            var textSize = tooltipTextComp.GetPreferredValues(tooltipText);
            const float textPadding = 4f;
            var backgroundSize = new Vector2(textSize.x + textPadding * 2f, textSize.y + textPadding * 2f);
            _tooltip.transform.Find("Background").GetComponent<RectTransform>().sizeDelta = backgroundSize;
            _tooltip.transform.SetAsLastSibling();
        }

        private void HideTooltip()
        {
            _tooltip.SetActive(false);
        }

        private void ShowGraph(List<int> valueList, IGraphVisual graphVisual)
        {
            foreach (var o in _gameObjects)
            {
                Destroy(o);
            }

            _gameObjects.Clear();
            _yLabelList.Clear();

            foreach (var graphVisualObject in _graphVisualObjectList)
            {
                graphVisualObject.CleanUp();
            }

            _graphVisualObjectList.Clear();

            var sizeDelta = _graphContainer.sizeDelta;
            float graphHeight = sizeDelta.y;
            _xSize = sizeDelta.x / valueList.Count;

            CalculateYScale(out var yMin, out var yMax);

            for (int i = 0; i < valueList.Count; i++)
            {
                float xPos = _xSize / 2 + i * _xSize;
                float yPos = (valueList[i] - yMin) / (yMax - yMin) * graphHeight;
                // var dashY = Instantiate(_dashTemplateX, _graphContainer, false);
                // dashY.gameObject.SetActive(true);
                // dashY.anchoredPosition = new Vector2(xPos, 0);
                // _gameObjects.Add(dashY.gameObject);

                var tooltipText = valueList[i].ToString();
                _graphVisualObjectList.Add(graphVisual.CreateGraphVisual(new Vector2(xPos, yPos), _xSize, tooltipText));
                pointList.Add(new Vector4(xPos, yPos, -5, 1));

                // var labelX = Instantiate(_labelTemplateX, _graphContainer, false);
                // labelX.gameObject.SetActive(true);
                // labelX.anchoredPosition = new Vector2(xPos, -20f);
                // labelX.GetComponent<TextMeshProUGUI>().text = i.ToString();
                // _gameObjects.Add(labelX.gameObject);
            }

            int valueCount = valueList.Count;
            bool valueIsLess = valueCount <= 15;
            int separatorCount = valueIsLess ? valueCount : 15;

            for (int i = 0; i < separatorCount; i++)
            {
                float normalizedValue = i / (float)separatorCount;
                float xPos = valueIsLess ? _xSize / 2 + i * _xSize : _xSize / 2 + normalizedValue * sizeDelta.x;
                string labelText = (normalizedValue * valueCount).ToString("F1");

                var dashY = Instantiate(_dashTemplateX, _graphContainer, false);
                dashY.gameObject.SetActive(true);
                dashY.anchoredPosition = new Vector2(xPos, 0);
                _gameObjects.Add(dashY.gameObject);

                var labelX = Instantiate(_labelTemplateX, _graphContainer, false);
                labelX.gameObject.SetActive(true);
                labelX.anchoredPosition = new Vector2(xPos, -25f);
                labelX.GetComponent<TextMeshProUGUI>().text = labelText;
                _gameObjects.Add(labelX.gameObject);
            }

            separatorCount = 10;
            for (int i = 0; i <= separatorCount; i++)
            {
                float normalizedValue = i / (float)separatorCount;

                var dashX = Instantiate(_dashTemplateY, _graphContainer, false);
                dashX.gameObject.SetActive(true);
                dashX.anchoredPosition = new Vector2(0, normalizedValue * graphHeight);
                _gameObjects.Add(dashX.gameObject);

                var labelY = Instantiate(_labelTemplateY, _graphContainer, false);
                labelY.gameObject.SetActive(true);
                labelY.anchoredPosition = new Vector2(-30f, normalizedValue * graphHeight);
                labelY.GetComponent<TextMeshProUGUI>().text = (yMin + normalizedValue * (yMax - yMin)).ToString("F1");
                _gameObjects.Add(labelY.gameObject);
                _yLabelList.Add(labelY);
            }
        }

        private void UpdateValue(int index, int value)
        {
            CalculateYScale(out var yMinBefore, out var yMaxBefore);
            _valueList[index] = value;
            CalculateYScale(out var yMin, out var yMax);

            bool yScaleChanged = Math.Abs(yMinBefore - yMin) > 0 || Math.Abs(yMaxBefore - yMax) > 0;
            float graphHeight = _graphContainer.sizeDelta.y;

            if (!yScaleChanged)
            {
                float xPos = _xSize + index * _xSize;
                float yPos = (_valueList[index] - yMin) / (yMax - yMin) * graphHeight;

                var tooltipText = value.ToString();
                _graphVisualObjectList[index].SetGraphVisualObjectInfo(new Vector2(xPos, yPos), _xSize, tooltipText);
                pointList[index] = new Vector4(xPos, yPos, -5, 1);
                return;
            }

            for (int i = 0; i < _valueList.Count; i++)
            {
                float xPos = _xSize + i * _xSize;
                float yPos = (_valueList[i] - yMin) / (yMax - yMin) * graphHeight;

                var tooltipText = _valueList[i].ToString();
                _graphVisualObjectList[i].SetGraphVisualObjectInfo(new Vector2(xPos, yPos), _xSize, tooltipText);
                pointList[i] = new Vector4(xPos, yPos, -5, 1);
            }

            for (int i = 0; i < _yLabelList.Count; i++)
            {
                float normalizedValue = i / (float)_yLabelList.Count;
                _yLabelList[i].GetComponent<TextMeshProUGUI>().text =
                    Mathf.RoundToInt(yMin + normalizedValue * (yMax - yMin)).ToString();
            }
        }

        private void CalculateYScale(out float yMin, out float yMax)
        {
            yMin = _valueList[0];
            yMax = _valueList[0];

            foreach (var value in _valueList)
            {
                if (value > yMax)
                    yMax = value;

                if (value < yMin)
                    yMin = value;
            }

            var buffer = (yMax - yMin) * 0.1f;
            yMax += buffer;
            yMin -= buffer;

            if (_startYScaleAtZero)
                yMin = 0;
        }

        private interface IGraphVisual
        {
            IGraphVisualObject CreateGraphVisual(Vector2 graphPos, float graphWith, string tooltipText);
        }

        private interface IGraphVisualObject
        {
            void SetGraphVisualObjectInfo(Vector2 graphPos, float graphWidth, string tooltipText);
            void CleanUp();
        }

        private class BarChartVisual : IGraphVisual
        {
            private readonly RectTransform _graphContainer;
            private readonly Color _barColor;
            private readonly float _barWithMult;

            public BarChartVisual(RectTransform graphContainer, Color barColor, float barWithMult)
            {
                _graphContainer = graphContainer;
                _barColor = barColor;
                _barWithMult = barWithMult;
            }

            public IGraphVisualObject CreateGraphVisual(Vector2 graphPos, float graphWith, string tooltipText)
            {
                var barGameObject = CreateBar(graphPos, graphWith);
                var barChartVisualObject = new BarChartVisualObject(barGameObject, _barWithMult);
                barChartVisualObject.SetGraphVisualObjectInfo(graphPos, graphWith, tooltipText);
                return barChartVisualObject;
            }

            private GameObject CreateBar(Vector2 graphPos, float barWith)
            {
                var bar = new GameObject("bar", typeof(Image), typeof(EventTrigger));
                bar.transform.SetParent(_graphContainer, false);
                bar.GetComponent<Image>().color = _barColor;

                var rectTransform = bar.GetComponent<RectTransform>();
                rectTransform.anchoredPosition = new Vector2(graphPos.x, 0f);
                rectTransform.sizeDelta = new Vector2(barWith * _barWithMult, graphPos.y);
                rectTransform.anchorMin = new Vector2(0, 0);
                rectTransform.anchorMax = new Vector2(0, 0);
                rectTransform.pivot = new Vector2(0.5f, 0f);
                return bar;
            }

            private class BarChartVisualObject : IGraphVisualObject
            {
                private readonly GameObject _barGameObject;
                private readonly float _barWidthMult;

                public BarChartVisualObject(GameObject barGameObject, float barWithMult)
                {
                    _barGameObject = barGameObject;
                    _barWidthMult = barWithMult;
                }

                public void SetGraphVisualObjectInfo(Vector2 graphPos, float graphWidth, string tooltipText)
                {
                    var rectTransform = _barGameObject.GetComponent<RectTransform>();
                    rectTransform.anchoredPosition = new Vector2(graphPos.x, 0f);
                    rectTransform.sizeDelta = new Vector2(graphWidth * _barWidthMult, graphPos.y);

                    var eventTriggerEnter = new EventTrigger.Entry
                    {
                        eventID = EventTriggerType.PointerEnter
                    };
                    eventTriggerEnter.callback.AddListener(_ => _instance.ShowTooltip(tooltipText, graphPos));

                    var eventTriggerExit = new EventTrigger.Entry
                    {
                        eventID = EventTriggerType.PointerExit
                    };
                    eventTriggerExit.callback.AddListener(_ => _instance.HideTooltip());

                    var trigger = _barGameObject.GetComponent<EventTrigger>();
                    trigger.triggers.Clear();
                    trigger.triggers.Add(eventTriggerEnter);
                    trigger.triggers.Add(eventTriggerExit);
                }

                public void CleanUp()
                {
                    Destroy(_barGameObject);
                }
            }
        }

        private class LineGraphVisual : IGraphVisual
        {
            private readonly RectTransform _graphContainer;
            private readonly Sprite _dotSprite;
            private LineGraphVisualObject _lastDot;
            private readonly Color _dotColor;
            private readonly Color _lineColor;

            public LineGraphVisual(RectTransform graphContainer, Sprite dotSprite, Color dotColor, Color lineColor)
            {
                _graphContainer = graphContainer;
                _dotSprite = dotSprite;
                _lastDot = null;
                _dotColor = dotColor;
                _lineColor = lineColor;
            }

            public IGraphVisualObject CreateGraphVisual(Vector2 graphPos, float graphWith, string tooltipText)
            {
                var dot = CreateDot(graphPos);

                GameObject dotConnection = null;
                if (_lastDot != null)
                {
                    dotConnection = CreateDotConnection(_lastDot.GetGraphPosition(),
                        dot.GetComponent<RectTransform>().anchoredPosition);
                }

                var lineGraphVisualObject = new LineGraphVisualObject(dot, dotConnection, _lastDot);
                lineGraphVisualObject.SetGraphVisualObjectInfo(graphPos, graphWith, tooltipText);
                _lastDot = lineGraphVisualObject;

                return lineGraphVisualObject;
            }

            private GameObject CreateDot(Vector2 anchoredPosition)
            {
                var dot = new GameObject("dot", typeof(Image));
                dot.transform.SetParent(_graphContainer, false);
                dot.GetComponent<Image>().sprite = _dotSprite;
                dot.GetComponent<Image>().color = /*_dotColor*/ new Color(0, 0, 0, 0);

                var rectTransform = dot.GetComponent<RectTransform>();
                rectTransform.anchoredPosition = anchoredPosition;
                rectTransform.sizeDelta = new Vector2(20, 20);
                rectTransform.anchorMin = new Vector2(0, 0);
                rectTransform.anchorMax = new Vector2(0, 0);
                return dot;
            }

            private GameObject CreateDotConnection(Vector2 dotPositionA, Vector2 dotPositionB)
            {
                var line = new GameObject("dotConnection", typeof(Image));
                line.transform.SetParent(_graphContainer, false);
                line.GetComponent<Image>().color = /*_lineColor*/ new Color(0, 0, 0, 0);
                line.GetComponent<Image>().raycastTarget = false;

                var dir = (dotPositionB - dotPositionA).normalized;
                var dist = (dotPositionB - dotPositionA).magnitude;

                var rectTransform = line.GetComponent<RectTransform>();
                rectTransform.anchoredPosition = dotPositionA + dir * (dist * 0.5f);
                rectTransform.sizeDelta = new Vector2(dist, 3f);
                rectTransform.anchorMin = new Vector2(0, 0);
                rectTransform.anchorMax = new Vector2(0, 0);

                var angle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;
                rectTransform.localEulerAngles = new Vector3(0, 0, angle);
                return line;
            }

            public class LineGraphVisualObject : IGraphVisualObject
            {
                private event EventHandler OnChangeGraphVisualObjectInfo;
                private readonly GameObject _dotGameObject;
                private readonly GameObject _dotConnectionGameObject;
                private readonly LineGraphVisualObject _lastDot;

                public LineGraphVisualObject(GameObject dotGameObject, GameObject dotConnectionGameObject,
                    LineGraphVisualObject lastDot)
                {
                    _dotGameObject = dotGameObject;
                    _dotConnectionGameObject = dotConnectionGameObject;
                    _lastDot = lastDot;

                    if (_lastDot != null)
                    {
                        _lastDot.OnChangeGraphVisualObjectInfo += OnOnChangeGraphVisualObjectInfo;
                    }
                }

                protected virtual void OnOnChangeGraphVisualObjectInfo(object sender, EventArgs args)
                {
                    UpdateDotConnection();
                }

                public void SetGraphVisualObjectInfo(Vector2 graphPos, float graphWidth, string tooltipText)
                {
                    var rectTransform = _dotGameObject.GetComponent<RectTransform>();
                    rectTransform.anchoredPosition = graphPos;
                    UpdateDotConnection();
                    OnChangeGraphVisualObjectInfo?.Invoke(this, EventArgs.Empty);
                }

                public Vector2 GetGraphPosition()
                {
                    var rectTransform = _dotGameObject.GetComponent<RectTransform>();
                    return rectTransform.anchoredPosition;
                }

                private void UpdateDotConnection()
                {
                    if (_dotConnectionGameObject == null) return;

                    var graphPosA = GetGraphPosition();
                    var graphPosB = _lastDot.GetGraphPosition();
                    var dir = (graphPosB - graphPosA).normalized;
                    var dist = (graphPosB - graphPosA).magnitude;
                    var angle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;

                    var rectTransform = _dotConnectionGameObject.GetComponent<RectTransform>();
                    rectTransform.anchoredPosition = graphPosA + dir * dist * 0.5f;
                    rectTransform.sizeDelta = new Vector2(dist, 3f);
                    rectTransform.localEulerAngles = new Vector3(0, 0, angle);
                }

                public void CleanUp()
                {
                    Destroy(_dotGameObject);
                    Destroy(_dotConnectionGameObject);
                }
            }
        }
    }
}