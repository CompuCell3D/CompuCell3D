

import CalculatorTry.*;
public class CalculatorPackageUser {
   public static void main  (String argv[]) {

//       Loader loader=new Loader();
//      String oldPath=System.getProperty("java.library.path");
//      String newPath=oldPath+System.getProperty("path.separator")+"c:\\Program Files\\classes\\CalculatorTry";
//      System.setProperty("java.library.path",newPath);
//      System.out.println("THIS IS PATH "+System.getProperty("java.library.path"));
// 
//      System.out.println("This is before loading classes library");
//      System.loadLibrary("ClassesLib");
     System.loadLibrary("classes");

//      loader.calculate();

      Calculator calculator=new Calculator();
      calculator.calculate(10);

   }
 }