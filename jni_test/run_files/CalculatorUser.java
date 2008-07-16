


public class CalculatorUser {
   public static void main  (String argv[]) {
     System.out.println("This is before loading classes library");


     System.loadLibrary("classes");
     Calculator calculator=new Calculator();
//      calculator.calculate(10);

   }
 }