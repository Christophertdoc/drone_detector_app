class BoundingBox {
  final double left;
  final double top;
  final double right;
  final double bottom;

  BoundingBox({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });
}

class Detection {
  final BoundingBox boundingBox;
  final double confidence;

  Detection({required this.boundingBox, required this.confidence});
}
