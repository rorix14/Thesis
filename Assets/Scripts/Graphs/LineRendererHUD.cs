using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace Graphs
{
    public class LineRendererHUD : Graphic
    {
        public float thickness;
        public List<Vector2> points;
        
        // cached variables 
        private float _unitWidth;
        private float _unitHeight;
        private Vector2 _initialLGridSize;
        
        protected override void Start()
        {
            base.Start();
            var rect = rectTransform.rect;
            _initialLGridSize = new Vector2(rect.width, rect.height);

            if(points != null) return;
            points = new List<Vector2>();
        }

        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear();

            if (points.Count < 2) return;

            var rect = rectTransform.rect;
            _unitWidth = rect.width / _initialLGridSize.x;
            _unitHeight = rect.height / _initialLGridSize.y;

            for (int i = 0; i < points.Count - 1; i++)
            {
                Vector2 point = points[i];
                Vector2 point2 = points[i + 1];

                var angle = GetAngle(point, point2) + 90f;
                DrawVerticesForPoint(point, point2, angle, vh);
                
                int index = i * 4;
                vh.AddTriangle(index + 0, index + 1, index + 2);
                vh.AddTriangle(index + 1, index + 2, index + 3);

                if (i >= points.Count - 2) continue;

                vh.AddTriangle(index + 2, index + 3, index + 4);
                vh.AddTriangle(index + 3, index + 4, index + 5);
            }
        }

        private float GetAngle(Vector2 me, Vector2 target)
        {
            return Mathf.Atan2(target.y - me.y, target.x - me.x) * Mathf.Rad2Deg;
        }

        private void DrawVerticesForPoint(Vector2 point, Vector2 point2, float angle, VertexHelper vh)
        {
            var vertex = UIVertex.simpleVert;
            vertex.color = color;

            vertex.position = Quaternion.Euler(0, 0, angle) * new Vector3(-thickness / 2, 0);
            vertex.position += new Vector3(_unitWidth * point.x, _unitHeight * point.y);
            vh.AddVert(vertex);

            vertex.position = Quaternion.Euler(0, 0, angle) * new Vector3(thickness / 2, 0);
            vertex.position += new Vector3(_unitWidth * point.x, _unitHeight * point.y);
            vh.AddVert(vertex);

            vertex.position = Quaternion.Euler(0, 0, angle) * new Vector3(-thickness / 2, 0);
            vertex.position += new Vector3(_unitWidth * point2.x, _unitHeight * point2.y);
            vh.AddVert(vertex);

            vertex.position = Quaternion.Euler(0, 0, angle) * new Vector3(thickness / 2, 0);
            vertex.position += new Vector3(_unitWidth * point2.x, _unitHeight * point2.y);
            vh.AddVert(vertex);
        }
    }
}