[CmdletBinding()]
param(
    [string]$Rater1Form = (Join-Path $PSScriptRoot '..\results\q1_1\review_intake\v1\rater_1_form.csv'),
    [string]$Rater2Form = (Join-Path $PSScriptRoot '..\results\q1_1\review_intake\v1\rater_2_form.csv'),
    [string]$OutputPath = (Join-Path $PSScriptRoot '..\results\q1_1\review_intake\v1\validated_ratings.csv')
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$rosterPath = Join-Path $projectRoot 'results\q1_1\v1\review_packet\blind_review_form.csv'
$expectedIds = @(Import-Csv -LiteralPath $rosterPath | ForEach-Object { $_.blind_id })
$criteria = @('education', 'readability', 'coherence', 'information', 'overall')

function Read-And-ValidateForm {
    param(
        [string]$Path,
        [string]$RaterPrefix,
        [string[]]$ExpectedIds,
        [string[]]$Criteria
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "找不到评分表：$Path"
    }

    $rows = @(Import-Csv -LiteralPath $Path)
    if ($rows.Count -eq 0) {
        throw "评分表为空：$Path"
    }
    $expectedColumns = @('blind_id') + @($Criteria | ForEach-Object { $RaterPrefix + '_' + $_ })
    $actualColumns = @($rows[0].PSObject.Properties.Name)
    if (($expectedColumns -join '|') -cne ($actualColumns -join '|')) {
        throw "评分表列名不符合要求：$Path；预期列为 $($expectedColumns -join ', ')"
    }

    $ids = @($rows | ForEach-Object { $_.blind_id })
    $duplicateIds = @($ids | Group-Object | Where-Object Count -gt 1 | ForEach-Object Name)
    $missingIds = @($ExpectedIds | Where-Object { $_ -notin $ids })
    $extraIds = @($ids | Where-Object { $_ -notin $ExpectedIds })
    if ($rows.Count -ne $ExpectedIds.Count -or $duplicateIds.Count -gt 0 -or
        $missingIds.Count -gt 0 -or $extraIds.Count -gt 0) {
        throw "ID 校验失败：$Path；行数=$($rows.Count)，预期=$($ExpectedIds.Count)，重复=$($duplicateIds.Count)，缺失=$($missingIds.Count)，多余=$($extraIds.Count)"
    }

    $invalid = [System.Collections.Generic.List[string]]::new()
    foreach ($row in $rows) {
        foreach ($criterion in $Criteria) {
            $column = $RaterPrefix + '_' + $criterion
            $raw = [string]$row.$column
            $score = 0
            if ([string]::IsNullOrWhiteSpace($raw) -or
                -not [int]::TryParse($raw.Trim(), [ref]$score) -or
                $score -lt 1 -or $score -gt 5) {
                $invalid.Add("$($row.blind_id):$column='$raw'")
            }
        }
    }
    if ($invalid.Count -gt 0) {
        $sample = @($invalid | Select-Object -First 10) -join '; '
        throw "评分值校验失败：$Path；空值/非整数/越界项共 $($invalid.Count)。前 10 项：$sample"
    }

    $byId = @{}
    foreach ($row in $rows) {
        $byId[$row.blind_id] = $row
    }
    return [pscustomobject]@{ Rows = $rows; ById = $byId }
}

$r1 = Read-And-ValidateForm -Path $Rater1Form -RaterPrefix 'rater_1' -ExpectedIds $expectedIds -Criteria $criteria
$r2 = Read-And-ValidateForm -Path $Rater2Form -RaterPrefix 'rater_2' -ExpectedIds $expectedIds -Criteria $criteria

if (Test-Path -LiteralPath $OutputPath) {
    throw "输出文件已存在，为避免覆盖请另选路径：$OutputPath"
}

$merged = foreach ($id in $expectedIds) {
    $record = [ordered]@{ blind_id = $id }
    foreach ($criterion in $criteria) {
        $column = 'rater_1_' + $criterion
        $record[$column] = $r1.ById[$id].$column
    }
    foreach ($criterion in $criteria) {
        $column = 'rater_2_' + $criterion
        $record[$column] = $r2.ById[$id].$column
    }
    [pscustomobject]$record
}

$parent = Split-Path -Parent $OutputPath
if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
    throw "输出目录不存在：$parent"
}
$merged | Export-Csv -LiteralPath $OutputPath -NoTypeInformation -Encoding UTF8
Write-Output "校验通过并合并 $($merged.Count) 条双评审记录：$OutputPath"
