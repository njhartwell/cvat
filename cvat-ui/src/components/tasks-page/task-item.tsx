// Copyright (C) 2020-2022 Intel Corporation
// Copyright (C) 2022-2023 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { RouteComponentProps } from 'react-router';
import { withRouter } from 'react-router-dom';
import Text from 'antd/lib/typography/Text';
import { Row, Col } from 'antd/lib/grid';
import Button from 'antd/lib/button';
import { MoreOutlined } from '@ant-design/icons';
import Dropdown from 'antd/lib/dropdown';
import Progress from 'antd/lib/progress';
import Badge from 'antd/lib/badge';
import moment from 'moment';

import ActionsMenuContainer from 'containers/actions-menu/actions-menu';
import Preview from 'components/common/preview';
import { ActiveInference, PluginComponent } from 'reducers';
import AutomaticAnnotationProgress from './automatic-annotation-progress';

export interface TaskItemProps {
    taskInstance: any;
    deleted: boolean;
    hidden: boolean;
    activeInference: ActiveInference | null;
    ribbonPlugins: PluginComponent[];
    cancelAutoAnnotation(): void;
}

class TaskItemComponent extends React.PureComponent<TaskItemProps & RouteComponentProps> {
    private renderPreview(): JSX.Element {
        const { taskInstance } = this.props;
        return (
            <Col span={4}>
                <Preview
                    task={taskInstance}
                    loadingClassName='cvat-task-item-loading-preview'
                    emptyPreviewClassName='cvat-task-item-empty-preview'
                    previewWrapperClassName='cvat-task-item-preview-wrapper'
                    previewClassName='cvat-task-item-preview'
                />
            </Col>
        );
    }

    private renderDescription(): JSX.Element {
        // Task info
        const { taskInstance } = this.props;
        const { id } = taskInstance;
        const owner = taskInstance.owner ? taskInstance.owner.username : null;
        const updated = moment(taskInstance.updatedDate).fromNow();
        const created = moment(taskInstance.createdDate).format('MMMM Do YYYY');

        // Get and truncate a task name
        const name = `${taskInstance.name.substring(0, 70)}${taskInstance.name.length > 70 ? '...' : ''}`;

        return (
            <Col span={10} className='cvat-task-item-description'>
                <Text strong type='secondary' className='cvat-item-task-id'>{`#${id}: `}</Text>
                <Text strong className='cvat-item-task-name'>
                    {name}
                </Text>
                <br />
                {owner && (
                    <>
                        <Text type='secondary'>{`Created ${owner ? `by ${owner}` : ''} on ${created}`}</Text>
                        <br />
                    </>
                )}
                <Text type='secondary'>{`Last updated ${updated}`}</Text>
            </Col>
        );
    }

    private renderProgress(): JSX.Element {
        const { taskInstance, activeInference, cancelAutoAnnotation } = this.props;
        // Count number of jobs and performed jobs
        const numOfJobs = taskInstance.progress.totalJobs;
        const numOfCompleted = taskInstance.progress.completedJobs;
        const numOfValidation = taskInstance.progress.validationJobs;
        const numOfAnnotation = taskInstance.progress.annotationJobs;

        // Progress appearance depends on number of jobs
        const jobsProgress = ((numOfCompleted + numOfValidation) * 100) / numOfJobs;

        return (
            <Col span={7}>
                <Row>
                    <Col span={24} className='cvat-task-item-progress-wrapper'>
                        <div>
                            { numOfCompleted > 0 && (
                                <Text strong className='cvat-task-completed-progress'>
                                    {`\u2022 ${numOfCompleted} done `}
                                </Text>
                            )}

                            { numOfValidation > 0 && (
                                <Text strong className='cvat-task-validation-progress'>
                                    {`\u2022 ${numOfValidation} on review `}
                                </Text>
                            )}

                            { numOfAnnotation > 0 && (
                                <Text strong className='cvat-task-annotation-progress'>
                                    {`\u2022 ${numOfAnnotation} annotating `}
                                </Text>
                            )}
                            <Text strong type='secondary'>
                                {`\u2022 ${numOfJobs} total`}
                            </Text>
                        </div>
                        <Progress
                            percent={jobsProgress}
                            success={{
                                percent: (numOfCompleted * 100) / numOfJobs,
                            }}
                            strokeColor='#1890FF'
                            showInfo={false}
                            strokeWidth={5}
                            size='small'
                        />
                    </Col>
                </Row>
                <AutomaticAnnotationProgress
                    activeInference={activeInference}
                    cancelAutoAnnotation={cancelAutoAnnotation}
                />
            </Col>
        );
    }

    private renderNavigation(): JSX.Element {
        const { taskInstance, history } = this.props;
        const { id } = taskInstance;

        const onViewAnalytics = (): void => {
            history.push(`/tasks/${taskInstance.id}/analytics`);
        };

        return (
            <Col span={3}>
                <Row justify='end'>
                    <Col>
                        <Button
                            className='cvat-item-open-task-button'
                            type='primary'
                            size='large'
                            ghost
                            href={`/tasks/${id}`}
                            onClick={(e: React.MouseEvent): void => {
                                e.preventDefault();
                                history.push(`/tasks/${id}`);
                            }}
                        >
                            Open
                        </Button>
                    </Col>
                </Row>
                <Row justify='end'>
                    <Dropdown overlay={(
                        <ActionsMenuContainer
                            taskInstance={taskInstance}
                            onViewAnalytics={onViewAnalytics}
                        />
                    )}
                    >
                        <Col className='cvat-item-open-task-actions'>
                            <Text className='cvat-text-color'>Actions</Text>
                            <MoreOutlined className='cvat-menu-icon' />
                        </Col>
                    </Dropdown>
                </Row>
            </Col>
        );
    }

    public render(): JSX.Element {
        const { deleted, hidden, ribbonPlugins } = this.props;
        const style = {};
        if (deleted) {
            (style as any).pointerEvents = 'none';
            (style as any).opacity = 0.5;
        }

        if (hidden) {
            (style as any).display = 'none';
        }

        const ribbonItems = ribbonPlugins
            .filter((plugin) => plugin.data.shouldBeRendered(this.props, this.state))
            .map((plugin) => ({ component: plugin.component, weight: plugin.data.weight }));

        return (
            <Badge.Ribbon
                style={{ visibility: ribbonItems.length ? 'visible' : 'hidden' }}
                className='cvat-task-item-ribbon'
                placement='start'
                text={(
                    <div>
                        {ribbonItems.sort((item1, item2) => item1.weight - item2.weight)
                            .map((item) => item.component).map((Component, index) => (
                                <Component key={index} targetProps={this.props} targetState={this.state} />
                            ))}
                    </div>
                )}
            >
                <Row className='cvat-tasks-list-item' justify='center' align='top' style={{ ...style }}>
                    {this.renderPreview()}
                    {this.renderDescription()}
                    {this.renderProgress()}
                    {this.renderNavigation()}
                </Row>
            </Badge.Ribbon>
        );
    }
}

export default withRouter(TaskItemComponent);
