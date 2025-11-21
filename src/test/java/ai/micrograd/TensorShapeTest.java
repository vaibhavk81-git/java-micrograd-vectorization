package ai.micrograd;

import ai.micrograd.tensor.Tensor;
import ai.micrograd.tensor.TensorOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for tensor shape validation in operations.
 */
class TensorShapeTest {
    
    @Test
    void testMatmulValidShapes() {
        Tensor a = new Tensor(2, 3, false);
        Tensor b = new Tensor(3, 4, false);
        
        Tensor c = TensorOps.matmul(a, b);
        
        assertEquals(2, c.rows());
        assertEquals(4, c.cols());
    }
    
    @Test
    void testMatmulInvalidShapes() {
        Tensor a = new Tensor(2, 3, false);
        Tensor b = new Tensor(4, 2, false);  // Incompatible: 3 != 4
        
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> TensorOps.matmul(a, b)
        );
        
        assertTrue(ex.getMessage().contains("Shape mismatch"));
        assertTrue(ex.getMessage().contains("matmul"));
    }
    
    @Test
    void testAddRowVectorValidShapes() {
        Tensor matrix = new Tensor(5, 3, false);
        Tensor rowVec = new Tensor(1, 3, false);
        
        Tensor result = TensorOps.addRowVector(matrix, rowVec);
        
        assertEquals(5, result.rows());
        assertEquals(3, result.cols());
    }
    
    @Test
    void testAddRowVectorInvalidRowCount() {
        Tensor matrix = new Tensor(5, 3, false);
        Tensor notRowVec = new Tensor(2, 3, false);  // Must be 1 row
        
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> TensorOps.addRowVector(matrix, notRowVec)
        );
        
        assertTrue(ex.getMessage().contains("must have 1 row"));
    }
    
    @Test
    void testAddRowVectorInvalidColCount() {
        Tensor matrix = new Tensor(5, 3, false);
        Tensor rowVec = new Tensor(1, 4, false);  // Wrong number of columns
        
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> TensorOps.addRowVector(matrix, rowVec)
        );
        
        assertTrue(ex.getMessage().contains("Column mismatch"));
    }
    
    @Test
    void testElementwiseAddValidShapes() {
        Tensor a = new Tensor(3, 4, false);
        Tensor b = new Tensor(3, 4, false);
        
        Tensor c = TensorOps.add(a, b);
        
        assertEquals(3, c.rows());
        assertEquals(4, c.cols());
    }
    
    @Test
    void testElementwiseAddInvalidShapes() {
        Tensor a = new Tensor(3, 4, false);
        Tensor b = new Tensor(3, 5, false);  // Different column count
        
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> TensorOps.add(a, b)
        );
        
        assertTrue(ex.getMessage().contains("Shape mismatch"));
    }
    
    @Test
    void testElementwiseMulInvalidShapes() {
        Tensor a = new Tensor(2, 3, false);
        Tensor b = new Tensor(3, 2, false);  // Transposed shape
        
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> TensorOps.mul(a, b)
        );
        
        assertTrue(ex.getMessage().contains("Shape mismatch"));
    }
    
    @Test
    void testSumAxis0() {
        Tensor x = new Tensor(4, 5, false);
        
        Tensor result = TensorOps.sum(x, 0);
        
        assertEquals(1, result.rows());
        assertEquals(5, result.cols());
    }
    
    @Test
    void testSumAxis1() {
        Tensor x = new Tensor(4, 5, false);
        
        Tensor result = TensorOps.sum(x, 1);
        
        assertEquals(4, result.rows());
        assertEquals(1, result.cols());
    }
    
    @Test
    void testSumInvalidAxis() {
        Tensor x = new Tensor(4, 5, false);
        
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> TensorOps.sum(x, 2)
        );
        
        assertTrue(ex.getMessage().contains("axis must be 0 or 1"));
    }
    
    @Test
    void testMeanAxis0() {
        Tensor x = new Tensor(4, 5, false);
        
        Tensor result = TensorOps.mean(x, 0);
        
        assertEquals(1, result.rows());
        assertEquals(5, result.cols());
    }
    
    @Test
    void testMeanAxis1() {
        Tensor x = new Tensor(4, 5, false);
        
        Tensor result = TensorOps.mean(x, 1);
        
        assertEquals(4, result.rows());
        assertEquals(1, result.cols());
    }
}

