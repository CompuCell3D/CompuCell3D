#include "Vector2D.h"

Vector2D::Vector2D(double x, double y) : x_(x), y_(y) {}

Vector2D Vector2D::operator+(const Vector2D &other) const {
    return Vector2D(x_ + other.x_, y_ + other.y_);
}

Vector2D Vector2D::operator-(const Vector2D &other) const {
    return Vector2D(x_ - other.x_, y_ - other.y_);
}

Vector2D Vector2D::operator*(double scale) const {
    return Vector2D(x_ * scale, y_ * scale);
}

double Vector2D::dot(const Vector2D &other) const {
    return x_ * other.x_ + y_ * other.y_;
}

double Vector2D::normSquared() const {
    return dot(*this);
}
