<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xs="http://www.w3.org/2001/XMLSchema">
<xsl:template match="/">
<html>
<head />
<style type="text/css">
@import url("report_table.css");
</style>

  <body title="">
     <xsl:for-each select="unittest-results">
     <h3>TestDate: <xsl:for-each select="@date_time">
            <xsl:value-of select="." />
            </xsl:for-each></h3>

     <p><table id="summaryTable" >
          <caption>Summary         </caption>
          <thead>
          <tr>
                    <th scope="col">Total number of tests</th>
                    <th scope="col">Number of failed tests</th>
                    <th scope="col">Failures</th>
                    <th scope="col">Time (seconds)</th>
          </tr>
       </thead>
       <tbody>
        <tr>
        <td>
        <xsl:for-each select="@tests">
        <xsl:value-of select="." />
        </xsl:for-each>
        </td>

        <td>
        <xsl:for-each select="@failedtests">
        <xsl:value-of select="." />
        </xsl:for-each>
        </td>
        
        <td>
        <xsl:for-each select="@failures">
        <xsl:value-of select="." />
        </xsl:for-each>
        </td>

        <td>
        <xsl:for-each select="@time">
        <xsl:value-of select="." />
        </xsl:for-each>
        </td>
        </tr>
       </tbody>
       </table>

     <xsl:for-each select="test">
     <xsl:if test="position( )=1">
          <table id="detailsTable" border="0" cellspacing="0" >
            <caption>Details</caption>
          <thead>
          <tr>
                    <th scope="col">Suite </th>
                    <th scope="col">Name  </th>
                    <th scope="col">Time  </th>
                    <th scope="col">Status</th>
                    <th scope="col">Info  </th>
          </tr>
       </thead>
       <tbody>
       <xsl:for-each select="../test">
           <tr>
           <td >
              <xsl:for-each select="@suite">
              <xsl:value-of select="." />
              </xsl:for-each>
         </td>
         <td >
              <xsl:for-each select="@name">
              <xsl:value-of select="." />
              </xsl:for-each>
        </td>
        <td >
              <xsl:for-each select="@time">
              <xsl:value-of select="." />
              </xsl:for-each>
        </td>
        <td >
            <xsl:if test="failure">
            FAIL
            </xsl:if>
            <xsl:if test="not(failure)">
            PASS
            </xsl:if>
        </td>
         <td>
            <xsl:if test="failure">
            <xsl:for-each select="failure">
              <xsl:for-each select="@message">
              <xsl:value-of select="." />
              </xsl:for-each>
              <br/>
            </xsl:for-each>
            </xsl:if>
            <xsl:if test="not(failure)">
            N/A
            </xsl:if>
        </td>
        </tr>
        </xsl:for-each>
      </tbody>
      </table>
      </xsl:if>
</xsl:for-each>
</p>
</xsl:for-each>
</body>
</html>
</xsl:template>
</xsl:stylesheet>

