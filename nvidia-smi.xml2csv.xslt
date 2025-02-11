<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:variable name="newline"><xsl:text>
</xsl:text></xsl:variable>
  <xsl:template match="/nvidia_smi">
    <xsl:text>timestamp,gpu_util,memory_util</xsl:text>
    <xsl:value-of select="$newline"/>
    <xsl:for-each select="nvidia_smi_log">
      <xsl:value-of select="timestamp"/>
      <xsl:text>,</xsl:text>
      <xsl:value-of select="gpu/utilization/gpu_util"/>
      <xsl:text>,</xsl:text>
      <xsl:value-of select="gpu/utilization/memory_util"/>
      <xsl:if test="position() != last()">
        <xsl:value-of select="$newline"/>
      </xsl:if>
    </xsl:for-each>
  </xsl:template>
</xsl:stylesheet>
