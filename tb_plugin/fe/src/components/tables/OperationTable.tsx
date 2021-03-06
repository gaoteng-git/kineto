/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { OperationTableData, OperationTableDataInner } from '../../api'
import { OperationGroupBy } from '../../constants/groupBy'
import { attachId, getCommonOperationColumns } from './common'
import { Table, TablePaginationConfig, TableProps } from 'antd'
import { makeExpandIcon } from './ExpandIcon'
import { CallStackTable } from './CallStackTable'

export interface IProps {
  data: OperationTableData
  run: string
  worker: string
  span: string
  groupBy: OperationGroupBy
}
const rowExpandable = (record: OperationTableDataInner) => record.has_call_stack
const expandIcon = makeExpandIcon<OperationTableDataInner>(
  'View CallStack',
  (record) => !record.has_call_stack
)
export const OperationTable = (props: IProps) => {
  const { data, run, worker, span, groupBy } = props

  const rows = React.useMemo(() => attachId(data), [data])

  const columns = React.useMemo(() => getCommonOperationColumns(rows), [rows])

  const [pageSize, setPageSize] = React.useState(30)
  const onShowSizeChange = (current: number, size: number) => {
    setPageSize(size)
  }

  const expandIconColumnIndex = columns.length
  const expandedRowRender = React.useCallback(
    (record: OperationTableDataInner) => (
      <CallStackTable
        data={record}
        run={run}
        worker={worker}
        span={span}
        groupBy={groupBy}
      />
    ),
    [run, worker, span, groupBy]
  )

  const expandable: TableProps<OperationTableDataInner>['expandable'] = React.useMemo(
    () => ({
      expandIconColumnIndex,
      expandIcon,
      expandedRowRender,
      rowExpandable
    }),
    [expandIconColumnIndex, expandedRowRender]
  )

  return (
    <Table
      size="small"
      bordered
      columns={columns}
      dataSource={rows}
      pagination={{
        pageSize,
        pageSizeOptions: ['10', '20', '30', '50', '100'],
        onShowSizeChange
      }}
      expandable={expandable}
    />
  )
}
