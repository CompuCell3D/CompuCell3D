#pragma once

class Vector2D {
public:
    Vector2D(double x, double y);

    Vector2D operator+(const Vector2D &other) const;
    Vector2D operator-(const Vector2D &other) const;
    Vector2D operator*(double scale) const;

    double dot(const Vector2D &other) const;
    double normSquared() const;

private:
    double x_;
    double y_;
};
