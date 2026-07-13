/*
 # Fixed-point arithmetic implementation

    Overloads arithmetic operators to keep code legible while performing
    fixed-point arithmetic with the correct bit width and Q width. It is
    greedy and will always use the maximum bit width and Q width of the
    operands, though a method to truncate the result to a desired bit width
    and Q width is provided.
 
 */

use std::ops::{Mul, Add, Sub, Div};

#[derive(Debug, Clone, Copy)]
pub struct FixedPoint {
    bit_width: usize,
    q_width: usize,
    value: i64,
}

impl Add for FixedPoint {
    type Output = Self;

    // Adds two FixedPoint numbers, ensuring they have the same q_width. The result will have the maximum bit_width of the two operands and the same q_width.
    fn add(self, rhs: Self) -> Self::Output {
        let new_q_width = self.q_width.max(rhs.q_width);
        let new_bit_width = self.bit_width.max(rhs.bit_width);
        let new_value: i64;

        if new_q_width > self.q_width {
            let lhs = self.value << (new_q_width - self.q_width);
            new_value = lhs + rhs.value;
        } else if new_q_width > rhs.q_width {
            let rhs_scaled = rhs.value << (new_q_width - rhs.q_width);
            new_value = self.value + rhs_scaled;
        } else {
            new_value = self.value + rhs.value;
        }
        
        
        Self {
            bit_width: new_bit_width,
            q_width: new_q_width,
            value: new_value,
        }
    }
}

impl Sub for FixedPoint {
    type Output = Self;

    // Subtracts two FixedPoint numbers, ensuring they have the same q_width. The result will have the maximum bit_width of the two operands and the same q_width.
    fn sub(self, rhs: Self) -> Self::Output {
        let new_q_width = self.q_width.max(rhs.q_width);
        let new_bit_width = self.bit_width.max(rhs.bit_width);
        let new_value: i64;

        if new_q_width > self.q_width {
            let lhs = self.value << (new_q_width - self.q_width);
            new_value = lhs - rhs.value;
        } else if new_q_width > rhs.q_width {
            let rhs_scaled = rhs.value << (new_q_width - rhs.q_width);
            new_value = self.value - rhs_scaled;
        } else {
            new_value = self.value - rhs.value;
        }

        Self {
            bit_width: new_bit_width,
            q_width: new_q_width,
            value: new_value,
        }
    }
}

impl Mul for FixedPoint {
    type Output = Self;

    // Multiplication without truncation, the result will have bit_width and q_width that are the sum of the two operands' bit_width and q_width.
    fn mul(self, rhs: Self) -> Self::Output {
        let new_bit_width = self.bit_width + rhs.bit_width;
        let new_q_width = self.q_width + rhs.q_width;
        let new_value = self.value * rhs.value;
        Self {
            bit_width: new_bit_width,
            q_width: new_q_width,
            value: new_value,
        }
    }
}

impl Div for FixedPoint {
    type Output = Self;

    // Division without truncation, the result will have bit_width and q_width that are the difference of the two operands' bit_width and q_width.
    fn div(self, rhs: Self) -> Self::Output {
        let new_bit_width = self.bit_width + rhs.bit_width;
        let new_q_width = self.q_width + rhs.q_width;

        let scale_factor = 1 << (new_q_width - self.q_width) as i64;
        let new_value = self.value * scale_factor / rhs.value;
        Self {
            bit_width: new_bit_width,
            q_width: new_q_width,
            value: new_value,
        }
    }
}

impl FixedPoint{
    pub fn new<T: Into<f64>>(bit_width: usize, q_width: usize, value: T) -> Self {
        let scale_factor = 1 << q_width;
        let fixed_value = (value.into() * scale_factor as f64).round() as i64;
        Self {
            bit_width,
            q_width,
            value: fixed_value,
        }
    }

    pub fn truncate(&mut self, new_bit_width: usize, new_q_width: usize) -> Self {
        let scale_factor = 1 << new_q_width;
        let new_value = (self.value * scale_factor as i64) >> self.q_width;
        self.bit_width = new_bit_width;
        self.q_width = new_q_width;
        self.value = new_value;
        *self
    }

    pub fn to_f32(&self) -> f32 {
        let scale_factor = 1 << self.q_width;
        let full_result: f64 = (self.value as f64) / (scale_factor as f64);
        full_result as f32
    }
}