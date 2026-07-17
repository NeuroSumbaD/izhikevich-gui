/*
 # Fixed-point arithmetic implementation

    Overloads arithmetic operators to keep code legible while performing
    fixed-point arithmetic with the correct bit width and Q width. It is
    greedy and will always use the maximum bit width and Q width of the
    operands, though a method to truncate the result to a desired bit width
    and Q width is provided.
 
 */

use std::ops::{Mul,
    Add,
    Sub,
    // Div
};

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
        let new_bit_width = self.bit_width.max(rhs.bit_width) + 1; // +1 for potential carry
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
        let new_bit_width = self.bit_width.max(rhs.bit_width) + 1; // +1 for potential borrow
        let new_q_width = self.q_width.max(rhs.q_width);
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
        let mut new_value: i128 = self.value as i128 * rhs.value as i128; // Use i128 to prevent overflow during multiplication

        let (min_raw, max_raw) = (-(1_i128 << (new_bit_width - 1)), (1_i128 << (new_bit_width - 1)) - 1);
        new_value = new_value.max(min_raw).min(max_raw);

        Self {
            bit_width: new_bit_width,
            q_width: new_q_width,
            value: new_value as i64, // Cast back to i64 after clamping
        }
    }
}

// TODO: Correctly implement fixed-point division (a suitable resultant bit width is not clear)
// impl Div for FixedPoint {
//     type Output = Self;

//     // Division with the maximum q_width of the two operands.
//     fn div(self, rhs: Self) -> Self::Output {
//         let new_bit_width = self.bit_width + rhs.bit_width;
//         let new_q_width = self.q_width.max(rhs.q_width);

//         let scale_factor = 1 << (new_q_width + rhs.q_width - self.q_width) as i64;
//         let new_value = self.value * scale_factor / rhs.value;
//         Self {
//             bit_width: new_bit_width,
//             q_width: new_q_width,
//             value: new_value,
//         }
//     }
// }

impl FixedPoint{
    pub fn new<T: Into<f64>>(bit_width: usize, q_width: usize, value: T) -> Self {
        let scale_factor = 1 << q_width;

        let lsb = 1.0 / scale_factor as f64;
        let max_value: f64 = ((1 << (bit_width - q_width - 1)) as f64) - lsb;
        let min_value: f64 = (-(1 << (bit_width - q_width - 1))).into();

        let value = value.into().max(min_value).min(max_value);

        let fixed_value = (value * scale_factor as f64).round() as i64;
        Self {
            bit_width,
            q_width,
            value: fixed_value,
        }
    }

    /// truncate a fixed-point number to a smaller bit width and Q width
    pub fn truncate(&mut self, new_bit_width: usize, new_q_width: usize) -> Self {
        if new_bit_width >= self.bit_width && new_q_width >= self.q_width {
            panic!("New bit width or Q width must be smaller than the current ones. Current: bit_width={}, q_width={}. New: bit_width={}, q_width={}", self.bit_width, self.q_width, new_bit_width, new_q_width);
        }
        let shift_amount = self.q_width - new_q_width;
        let new_value = self.value >> shift_amount;
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