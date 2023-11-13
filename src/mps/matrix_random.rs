use super::*;
use crate::mps::matrix_multiplication::MatrixRef;
use std::ops::Add;

#[repr(C)]
pub enum MatrixRandomDistribution {
    Default = 1,
    Uniform = 2,
    Normal = 3,
}

/// Describes properties of a distribution of random values.
///
/// See <https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixrandomdistributiondescriptor?language=objc>
pub enum MPSMatrixRandomDistributionDescriptor {}
foreign_obj_type! {
    type CType = MPSMatrixRandomDistributionDescriptor;
    pub struct MatrixRandomDistributionDescriptor;
    type ParentType = NsObject;
}

impl MatrixRandomDistributionDescriptor {
    pub fn default() -> Self {
        unsafe {
            let descriptor: MatrixRandomDistributionDescriptor = msg_send![
                class!(MPSMatrixRandomDistributionDescriptor),
                defaultDistributionDescriptor
            ];
            descriptor
        }
    }

    pub fn uniform_distribution(minimum: f32, maximum: f32) -> Self {
        unsafe {
            let descriptor: MatrixRandomDistributionDescriptor = msg_send![
                class!(MPSMatrixRandomDistributionDescriptor),
                    uniformDistributionDescriptorWithMinimum : minimum
                                                     maximum : maximum
            ];
            descriptor
        }
    }

    pub fn normal_distribution(mean: f32, standard_deviation: f32) -> Self {
        unsafe {
            let descriptor: MatrixRandomDistributionDescriptor = msg_send![
                class!(MPSMatrixRandomDistributionDescriptor),
                        normalDistributionDescriptorWithMean : mean
                                           standardDeviation : standard_deviation
            ];
            descriptor
        }
    }

    pub fn normal_distribution_truncated(
        mean: f32,
        standard_deviation: f32,
        minimum: f32,
        maximum: f32,
    ) -> Self {
        unsafe {
            let descriptor: MatrixRandomDistributionDescriptor = msg_send![
                class!(MPSMatrixRandomDistributionDescriptor),
                        normalDistributionDescriptorWithMean : mean
                                           standardDeviation : standard_deviation
                                                     minimum : minimum
                                                     maximum : maximum
            ];
            descriptor
        }
    }
}

impl MatrixRandomDistributionDescriptorRef {
    pub fn distribution_type(&self) -> MatrixRandomDistribution {
        unsafe { msg_send![self, distributionType] }
    }

    pub fn set_distribution_type(&self, distribution_type: MatrixRandomDistribution) {
        unsafe { msg_send![self, setDistributionType : distribution_type] }
    }

    pub fn minimum(&self) -> f32 {
        unsafe { msg_send![self, minimum] }
    }

    pub fn set_minimum(&self, minimum: f32) {
        unsafe { msg_send![self, setMinimum : minimum] }
    }

    pub fn maximum(&self) -> f32 {
        unsafe { msg_send![self, maximum] }
    }

    pub fn set_maximum(&self, maximum: f32) {
        unsafe { msg_send![self, setMaximum : maximum] }
    }

    pub fn mean(&self) -> f32 {
        unsafe { msg_send![self, mean] }
    }

    pub fn set_mean(&self, mean: f32) {
        unsafe { msg_send![self, setMean : mean] }
    }

    pub fn standard_deviation(&self) -> f32 {
        unsafe { msg_send![self, standardDeviation] }
    }

    pub fn set_standard_deviation(&self, standard_deviation: f32) {
        unsafe { msg_send![self, setStandardDeviation : standard_deviation] }
    }
}

/// Kernels that implement random number generation.
///
/// See <https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixrandom?language=objc>
pub enum MPSMatrixRandom {}
foreign_obj_type! {
    type CType = MPSMatrixRandom;
    pub struct MatrixRandom;
    type ParentType = Kernel;
}

#[repr(C)]
pub enum DestinationDataType {
    UInt32 = UInt32::TYPE_ID as isize,
    Float32 = Float32::TYPE_ID as isize,
}

/// Marks a type as being suitable for use as a destination data type for
/// random number generation.
pub trait DestinationType: Clone + Copy + PartialEq + Debug + Add + PartialOrd + Into<f64> {
    const TYPE_ID: u32;
}
impl DestinationType for u32 {
    const TYPE_ID: u32 = UInt32::TYPE_ID;
}
impl DestinationType for f32 {
    const TYPE_ID: u32 = Float32::TYPE_ID;
}

/// Generates random numbers using a Mersenne Twister algorithm
/// suitable for GPU execution.  It uses a period of 2**11214.
///
/// For further details see:
/// Mutsuo Saito. A Variant of Mersenne Twister Suitable for Graphic Processors. arXiv:1005.4973
///
/// See <https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixrandommtgp32?language=objc>
pub enum MPSMatrixRandomMTGP32 {}

foreign_obj_type! {
    type CType = MPSMatrixRandomMTGP32;
    pub struct MatrixRandomMTGP32;
    type ParentType = MatrixRandom;
}

impl MatrixRandomMTGP32 {
    /// Initialize a MPSMatrixRandomMTGP32 filter to generate 32-bit unsigned
    /// integer values with an initial seed of 0.
    pub fn default(device: &DeviceRef) -> Option<Self> {
        Self::init_with_descriptor::<u32>(device, 0, &MatrixRandomDistributionDescriptor::default())
    }

    /// Initialize a MPSMatrixRandomMTGP32 filter using a default distribution.
    pub fn default_distribution<T: DestinationType>(
        device: &DeviceRef,
        seed: NSUInteger,
    ) -> Option<Self> {
        Self::init_with_descriptor::<T>(
            device,
            seed,
            &MatrixRandomDistributionDescriptor::default(),
        )
    }

    /// Initialize a MPSMatrixRandomMTGP32 filter.
    pub fn init_with_descriptor<T: DestinationType>(
        device: &DeviceRef,
        seed: NSUInteger,
        descriptor: &MatrixRandomDistributionDescriptorRef,
    ) -> Option<Self> {
        unsafe {
            let kernel: MatrixRandomMTGP32 = msg_send![class!(MPSMatrixRandomMTGP32), alloc];
            let ptr: *mut Object = msg_send![kernel.as_ref(),
                                             initWithDevice : device
                                        destinationDataType : T::TYPE_ID
                                                       seed : seed
                                     distributionDescriptor : descriptor
            ];
            let () = msg_send![descriptor, retain];
            if ptr.is_null() {
                None
            } else {
                Some(kernel)
            }
        }
    }

    /*
    TODO: Requires implementing NSCoder
    init_with_coder_device(aDecoder: NSCoder, device: id <MTLDevice>) -> Option<Self> {
        unsafe {
            let kernel: MatrixRandomMTGP32 = msg_send![class!(MatrixRandomMTGP32), alloc];
            let ptr: *mut Object = msg_send![kernel.as_ref(),
                                              initWithCoder : aDecoder
                                                     device : device
            ];
            if ptr.is_null() {
                None
            } else {
                Some(kernel)
            }
        }
     */
}

impl MatrixRandomMTGP32Ref {
    pub fn destination_data_type(&self) -> DestinationDataType {
        unsafe { msg_send![self, destinationDataType] }
    }

    pub fn distribution_type(&self) -> MatrixRandomDistribution {
        unsafe { msg_send![self, distributionType] }
    }

    pub fn batch_start(&self) -> NSUInteger {
        unsafe { msg_send![self, batchStart] }
    }

    pub fn set_batch_start(&self, batch_start: NSUInteger) {
        unsafe { msg_send![self, setBatchStart : batch_start] }
    }

    pub fn batch_size(&self) -> NSUInteger {
        unsafe { msg_send![self, batchSize] }
    }

    pub fn set_batch_size(&self, batch_size: NSUInteger) {
        unsafe { msg_send![self, setBatchSize : batch_size] }
    }

    /*
    TODO: Requires implementing Vector
    pub fn encode_into_vector(
        &self,
        command_buffer: &CommandBufferRef,
        destination_vector: &VectorRef,
    ) {
        unsafe {
            msg_send![self, encodeToCommandBuffer: command_buffer
                                destinationVector: destination_vector]
        }
    }
    */

    pub fn encode_into_matrix(
        &self,
        command_buffer: &CommandBufferRef,
        destination_matrix: &mut MatrixRef,
    ) {
        unsafe {
            msg_send![self, encodeToCommandBuffer : command_buffer
                                destinationMatrix : destination_matrix]
        }
    }

    /// Synchronize internal MTGP32 state between GPU and CPU.
    pub fn synchronize_state_on_command_buffer(&self, command_buffer: &CommandBufferRef) {
        unsafe { msg_send![self, synchronizeStateOnCommandBuffer : command_buffer] }
    }
}

/// Generates random numbers using a counter based algorithm.
///
/// For further details see:
/// John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw. Parallel Random Numbers: As Easy as 1, 2, 3.
///
/// See <https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixrandomphilox?language=objc>
pub enum MPSMatrixRandomPhilox {}
foreign_obj_type! {
    type CType = MPSMatrixRandomPhilox;
    pub struct MatrixRandomPhilox;
    type ParentType = MatrixRandom;
}

impl MatrixRandomPhilox {
    /// Initialize a MPSMatrixRandomPhilox filter to generate 32-bit unsigned
    /// integer values with an initial seed of 0.
    pub fn default(device: &DeviceRef) -> Option<Self> {
        Self::init_with_descriptor::<u32>(device, 0, &MatrixRandomDistributionDescriptor::default())
    }

    /// Initialize a MPSMatrixRandomPhilox filter using a default distribution.
    pub fn default_distribution<T: DestinationType>(
        device: &DeviceRef,
        seed: NSUInteger,
    ) -> Option<Self> {
        Self::init_with_descriptor::<T>(
            device,
            seed,
            &MatrixRandomDistributionDescriptor::default(),
        )
    }

    /// Initialize a MPSMatrixRandomPhilox filter
    pub fn init_with_descriptor<T: DestinationType>(
        device: &DeviceRef,
        seed: NSUInteger,
        descriptor: &MatrixRandomDistributionDescriptorRef,
    ) -> Option<Self> {
        unsafe {
            let kernel: MatrixRandomPhilox = msg_send![class!(MPSMatrixRandomPhilox), alloc];
            let ptr: *mut Object = msg_send![kernel.as_ref(),
                                             initWithDevice : device
                                        destinationDataType : T::TYPE_ID
                                                       seed : seed
                                     distributionDescriptor : descriptor
            ];
            let () = msg_send![descriptor, retain];
            if ptr.is_null() {
                None
            } else {
                Some(kernel)
            }
        }
    }

    /*
    TODO: Requires implementing NSCoder
    pub fn init_with_coder(decoder: NSCoder, device: id <MTLDevice>) -> Option<Self> {
        unsafe {
            let kernel: MatrixRandomPhilox = msg_send![class!(MatrixRandomPhilox), alloc];
            let ptr: *mut Object = msg_send![kernel.as_ref(),
                                              initWithCoder : decoder
                                                     device : device
            ];
            if ptr.is_null() {
                None
            } else {
                Some(kernel)
            }
        }
    }
    */
}

impl MatrixRandomPhiloxRef {
    pub fn destination_data_type(&self) -> DestinationDataType {
        unsafe { msg_send![self, destinationDataType] }
    }

    pub fn distribution_type(&self) -> MatrixRandomDistribution {
        unsafe { msg_send![self, distributionType] }
    }

    pub fn batch_start(&self) -> NSUInteger {
        unsafe { msg_send![self, batchStart] }
    }

    pub fn set_batch_start(&self, batch_start: NSUInteger) {
        unsafe { msg_send![self, setBatchStart : batch_start] }
    }

    pub fn batch_size(&self) -> NSUInteger {
        unsafe { msg_send![self, batchSize] }
    }

    pub fn set_batch_size(&self, batch_size: NSUInteger) {
        unsafe { msg_send![self, setBatchSize : batch_size] }
    }

    /*
    TODO: Requires implementing Vector
    pub fn encode_into_vector(
        &self,
        command_buffer: &CommandBufferRef,
        destination_vector: &VectorRef,
    ) {
        unsafe {
            msg_send![self, encodeToCommandBuffer: command_buffer
                                destinationVector: destination_vector]
        }
    }
    */

    pub fn encode_into_matrix(
        &self,
        command_buffer: &CommandBufferRef,
        destination_matrix: &MatrixRef,
    ) {
        unsafe {
            msg_send![self, encodeToCommandBuffer : command_buffer
                                destinationMatrix : destination_matrix]
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::mps::matrix_multiplication::{Matrix, MatrixDescriptor};
    use crate::mps::matrix_random::*;
    use crate::Device;
    use std::iter::Sum;

    fn mean<T>(values: Vec<T>) -> f64
    where
        T: Into<f64> + Sum<T> + Clone,
    {
        let len = values.len() as f64;
        let mut sum = 0.0;
        values
            .clone()
            .into_iter()
            .for_each(|v| sum += v.into() / len);

        sum
    }

    fn verify_matrix_random<T: DestinationType + Sum<T>>(
        rows: NSUInteger,
        columns: NSUInteger,
        seed: NSUInteger,
    ) {
        let device = Device::system_default().expect("No device found");
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let bytes_for_columns = MatrixDescriptor::row_bytes_for_columns(columns, T::TYPE_ID);
        let buffer = device.new_buffer(
            rows * bytes_for_columns,
            MTLResourceOptions::StorageModeShared,
        );
        let descriptor =
            MatrixDescriptor::init_single(rows, columns, bytes_for_columns, T::TYPE_ID);

        let mut matrix = Matrix::init_with_buffer_descriptor(&buffer, 0, &descriptor)
            .ok_or_else(|| "Failed to create left matrix")
            .unwrap();

        let mean_val = 10.0;
        let std_val = 1.0;
        let rand_descriptor = MatrixRandomDistributionDescriptor::normal_distribution_truncated(
            mean_val, std_val, -10.0, 10.0,
        );
        let matrix_random =
            MatrixRandomMTGP32::init_with_descriptor::<T>(&device, seed, &rand_descriptor)
                .ok_or_else(|| "Failed to create MatrixRandomMTGP32")
                .unwrap();

        matrix_random.synchronize_state_on_command_buffer(&command_buffer);
        matrix_random.encode_into_matrix(command_buffer, &mut matrix);
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let size = rows * columns;
        let result = buffer.read_to_vec::<T>(size as usize);
        println!("Result: {:?}", result);
        assert_eq!(result.len(), size as usize);
        assert!(mean(result) > 0.0);
        assert_eq!(true, false);
    }

    fn verify_matrix_random2<T: DestinationType + Sum<T>>(
        rows: NSUInteger,
        columns: NSUInteger,
        seed: NSUInteger,
    ) {
        let device = Device::system_default().expect("No device found");
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let bytes_for_columns = MatrixDescriptor::row_bytes_for_columns(columns, T::TYPE_ID);
        let buffer = device.new_buffer(
            rows * bytes_for_columns,
            MTLResourceOptions::StorageModeShared,
        );
        let descriptor =
            MatrixDescriptor::init_single(rows, columns, bytes_for_columns, T::TYPE_ID);

        let mut matrix = Matrix::init_with_buffer_descriptor(&buffer, 0, &descriptor)
            .ok_or_else(|| "Failed to create left matrix")
            .unwrap();

        let mean_val = 10.0;
        let std_val = 1.0;
        let rand_descriptor = MatrixRandomDistributionDescriptor::normal_distribution_truncated(
            mean_val, std_val, -10.0, 10.0,
        );
        let matrix_random =
            MatrixRandomPhilox::init_with_descriptor::<T>(&device, seed, &rand_descriptor)
                .ok_or_else(|| "Failed to create MatrixRandomMTGP32")
                .unwrap();

        matrix_random.encode_into_matrix(command_buffer, &mut matrix);
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let size = rows * columns;
        let result = buffer.read_to_vec::<T>(size as usize);
        println!("Result: {:?}", result);
        assert_eq!(result.len(), size as usize);
        assert!(mean(result) > 0.0);
        assert_eq!(true, false);
    }

    #[test]
    pub fn verify_MTGP32() {
        const R: NSUInteger = 5;
        const C: NSUInteger = 5;

        verify_matrix_random::<u32>(R, C, 23);
        // Float32
        verify_matrix_random::<f32>(R, C, 23);
    }

    #[test]
    pub fn verify_philox() {
        const R: NSUInteger = 5;
        const C: NSUInteger = 5;

        // UInt32
        verify_matrix_random2::<u32>(R, C, 23);
        // // Float32
        verify_matrix_random2::<f32>(R, C, 23);
    }
}
