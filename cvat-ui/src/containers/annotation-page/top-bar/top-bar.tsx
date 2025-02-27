// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) 2023 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { connect } from 'react-redux';
import { withRouter } from 'react-router';
import { RouteComponentProps } from 'react-router-dom';
import Input from 'antd/lib/input';
import copy from 'copy-to-clipboard';

import {
    changeFrameAsync,
    changeWorkspace as changeWorkspaceAction,
    collectStatisticsAsync,
    redoActionAsync,
    saveAnnotationsAsync,
    searchAnnotationsAsync,
    searchEmptyFrameAsync,
    setForceExitAnnotationFlag as setForceExitAnnotationFlagAction,
    showFilters as showFiltersAction,
    showStatistics as showStatisticsAction,
    switchPlay,
    undoActionAsync,
    deleteFrameAsync,
    restoreFrameAsync,
    switchNavigationBlocked as switchNavigationBlockedAction,
} from 'actions/annotation-actions';
import AnnotationTopBarComponent from 'components/annotation-page/top-bar/top-bar';
import { Canvas } from 'cvat-canvas-wrapper';
import { Canvas3d } from 'cvat-canvas3d-wrapper';
import { DimensionType, JobType } from 'cvat-core-wrapper';
import {
    CombinedState,
    FrameSpeed,
    Workspace,
    ActiveControl,
    ToolsBlockerState,
} from 'reducers';
import isAbleToChangeFrame from 'utils/is-able-to-change-frame';
import GlobalHotKeys, { KeyMap } from 'utils/mousetrap-react';
import { switchToolsBlockerState } from 'actions/settings-actions';

interface StateToProps {
    jobInstance: any;
    frameIsDeleted: boolean;
    frameNumber: number;
    frameFilename: string;
    frameStep: number;
    frameSpeed: FrameSpeed;
    frameDelay: number;
    frameFetching: boolean;
    playing: boolean;
    saving: boolean;
    canvasIsReady: boolean;
    undoAction?: string;
    redoAction?: string;
    autoSave: boolean;
    autoSaveInterval: number;
    toolsBlockerState: ToolsBlockerState;
    showDeletedFrames: boolean;
    workspace: Workspace;
    keyMap: KeyMap;
    normalizedKeyMap: Record<string, string>;
    canvasInstance: Canvas | Canvas3d;
    forceExit: boolean;
    ranges: string;
    activeControl: ActiveControl;
}

interface DispatchToProps {
    onChangeFrame(frame: number, fillBuffer?: boolean, frameStep?: number): void;
    onSwitchPlay(playing: boolean): void;
    onSaveAnnotation(sessionInstance: any): void;
    showStatistics(sessionInstance: any): void;
    showFilters(sessionInstance: any): void;
    undo(sessionInstance: any, frameNumber: any): void;
    redo(sessionInstance: any, frameNumber: any): void;
    searchAnnotations(sessionInstance: any, frameFrom: number, frameTo: number): void;
    searchEmptyFrame(sessionInstance: any, frameFrom: number, frameTo: number): void;
    setForceExitAnnotationFlag(forceExit: boolean): void;
    changeWorkspace(workspace: Workspace): void;
    onSwitchToolsBlockerState(toolsBlockerState: ToolsBlockerState): void;
    deleteFrame(frame: number): void;
    restoreFrame(frame: number): void;
    switchNavigationBlocked(blocked: boolean): void;
}

function mapStateToProps(state: CombinedState): StateToProps {
    const {
        annotation: {
            player: {
                playing,
                ranges,
                frame: {
                    data: { deleted: frameIsDeleted },
                    filename: frameFilename,
                    number: frameNumber,
                    delay: frameDelay,
                    fetching: frameFetching,
                },
            },
            annotations: {
                saving: { uploading: saving, forceExit },
                history,
            },
            job: { instance: jobInstance },
            canvas: { ready: canvasIsReady, instance: canvasInstance, activeControl },
            workspace,
        },
        settings: {
            player: { frameSpeed, frameStep, showDeletedFrames },
            workspace: {
                autoSave,
                autoSaveInterval,
                toolsBlockerState,
            },
        },
        shortcuts: { keyMap, normalizedKeyMap },
    } = state;

    return {
        frameIsDeleted,
        frameStep,
        frameSpeed,
        frameDelay,
        frameFetching,
        playing,
        canvasIsReady,
        saving,
        frameNumber,
        frameFilename,
        jobInstance,
        undoAction: history.undo.length ? history.undo[history.undo.length - 1][0] : undefined,
        redoAction: history.redo.length ? history.redo[history.redo.length - 1][0] : undefined,
        autoSave,
        autoSaveInterval,
        toolsBlockerState,
        showDeletedFrames,
        workspace,
        keyMap,
        normalizedKeyMap,
        canvasInstance,
        forceExit,
        activeControl,
        ranges,
    };
}

function mapDispatchToProps(dispatch: any): DispatchToProps {
    return {
        onChangeFrame(frame: number, fillBuffer?: boolean, frameStep?: number): void {
            dispatch(changeFrameAsync(frame, fillBuffer, frameStep));
        },
        onSwitchPlay(playing: boolean): void {
            dispatch(switchPlay(playing));
        },
        onSaveAnnotation(sessionInstance: any): void {
            dispatch(saveAnnotationsAsync(sessionInstance));
        },
        showStatistics(sessionInstance: any): void {
            dispatch(collectStatisticsAsync(sessionInstance));
            dispatch(showStatisticsAction(true));
        },
        showFilters(): void {
            dispatch(showFiltersAction(true));
        },
        undo(sessionInstance: any, frameNumber: any): void {
            dispatch(undoActionAsync(sessionInstance, frameNumber));
        },
        redo(sessionInstance: any, frameNumber: any): void {
            dispatch(redoActionAsync(sessionInstance, frameNumber));
        },
        searchAnnotations(sessionInstance: any, frameFrom: number, frameTo: number): void {
            dispatch(searchAnnotationsAsync(sessionInstance, frameFrom, frameTo));
        },
        searchEmptyFrame(sessionInstance: any, frameFrom: number, frameTo: number): void {
            dispatch(searchEmptyFrameAsync(sessionInstance, frameFrom, frameTo));
        },
        changeWorkspace(workspace: Workspace): void {
            dispatch(changeWorkspaceAction(workspace));
        },
        setForceExitAnnotationFlag(forceExit: boolean): void {
            dispatch(setForceExitAnnotationFlagAction(forceExit));
        },
        onSwitchToolsBlockerState(toolsBlockerState: ToolsBlockerState): void {
            dispatch(switchToolsBlockerState(toolsBlockerState));
        },
        deleteFrame(frame: number): void {
            dispatch(deleteFrameAsync(frame));
        },
        restoreFrame(frame: number): void {
            dispatch(restoreFrameAsync(frame));
        },
        switchNavigationBlocked(blocked: boolean): void {
            dispatch(switchNavigationBlockedAction(blocked));
        },
    };
}

interface State {
    prevButtonType: 'regular' | 'filtered' | 'empty';
    nextButtonType: 'regular' | 'filtered' | 'empty';
}

type Props = StateToProps & DispatchToProps & RouteComponentProps;
class AnnotationTopBarContainer extends React.PureComponent<Props, State> {
    private inputFrameRef: React.RefObject<Input>;
    private autoSaveInterval: number | undefined;
    private unblock: any;

    constructor(props: Props) {
        super(props);
        this.inputFrameRef = React.createRef<Input>();
        this.state = {
            prevButtonType: 'regular',
            nextButtonType: 'regular',
        };
    }

    public componentDidMount(): void {
        const {
            autoSaveInterval, history, jobInstance, setForceExitAnnotationFlag,
        } = this.props;
        this.autoSaveInterval = window.setInterval(this.autoSave.bind(this), autoSaveInterval);

        // eslint-disable-next-line @typescript-eslint/no-this-alias
        const self = this;
        this.unblock = history.block((location: any) => {
            const { forceExit } = self.props;
            const { id: jobID, taskId: taskID } = jobInstance;

            if (
                jobInstance.annotations.hasUnsavedChanges() &&
                location.pathname !== `/tasks/${taskID}/jobs/${jobID}` &&
                !forceExit
            ) {
                return 'You have unsaved changes, please confirm leaving this page.';
            }

            if (forceExit) {
                setForceExitAnnotationFlag(false);
            }

            return undefined;
        });

        window.addEventListener('beforeunload', this.beforeUnloadCallback);
    }

    public componentDidUpdate(prevProps: Props): void {
        const { autoSaveInterval } = this.props;

        if (autoSaveInterval !== prevProps.autoSaveInterval) {
            if (this.autoSaveInterval) window.clearInterval(this.autoSaveInterval);
            this.autoSaveInterval = window.setInterval(this.autoSave.bind(this), autoSaveInterval);
        }
        this.play();
    }

    public componentWillUnmount(): void {
        window.clearInterval(this.autoSaveInterval);
        window.removeEventListener('beforeunload', this.beforeUnloadCallback);
        this.unblock();
    }

    private undo = (): void => {
        const { undo, jobInstance, frameNumber } = this.props;

        if (isAbleToChangeFrame()) {
            undo(jobInstance, frameNumber);
        }
    };

    private redo = (): void => {
        const { redo, jobInstance, frameNumber } = this.props;

        if (isAbleToChangeFrame()) {
            redo(jobInstance, frameNumber);
        }
    };

    private showStatistics = (): void => {
        const { jobInstance, showStatistics } = this.props;
        showStatistics(jobInstance);
    };

    private showFilters = (): void => {
        const { jobInstance, showFilters } = this.props;
        showFilters(jobInstance);
    };

    private onSwitchPlay = (): void => {
        const {
            frameNumber, jobInstance, onSwitchPlay, playing,
        } = this.props;

        if (playing) {
            onSwitchPlay(false);
        } else if (frameNumber < jobInstance.stopFrame) {
            onSwitchPlay(true);
        }
    };

    private onFirstFrame = async (): Promise<void> => {
        const {
            frameNumber, jobInstance, playing, onSwitchPlay, showDeletedFrames,
        } = this.props;

        const newFrame =
            await jobInstance.frames.search({ notDeleted: !showDeletedFrames }, jobInstance.startFrame, frameNumber);
        if (newFrame !== frameNumber && newFrame !== null) {
            if (playing) {
                onSwitchPlay(false);
            }
            this.changeFrame(newFrame);
        }
    };

    private onBackward = async (): Promise<void> => {
        const {
            frameNumber, frameStep, jobInstance, playing, onSwitchPlay, showDeletedFrames,
        } = this.props;

        const newFrame = await jobInstance.frames.search(
            { notDeleted: !showDeletedFrames, offset: frameStep },
            Math.max(jobInstance.startFrame, frameNumber - 1),
            jobInstance.startFrame,
        );

        if (newFrame !== frameNumber && newFrame !== null) {
            if (playing) {
                onSwitchPlay(false);
            }
            this.changeFrame(newFrame);
        }
    };

    private onPrevFrame = async (): Promise<void> => {
        const { prevButtonType } = this.state;
        const {
            frameNumber, jobInstance, playing, onSwitchPlay, showDeletedFrames,
        } = this.props;
        const { startFrame } = jobInstance;

        const frameFrom = Math.max(jobInstance.startFrame, frameNumber - 1);
        const newFrame = await jobInstance.frames.search(
            { notDeleted: !showDeletedFrames },
            frameFrom,
            jobInstance.startFrame,
        );

        if (newFrame !== frameNumber && newFrame !== null) {
            if (playing) {
                onSwitchPlay(false);
            }

            if (prevButtonType === 'regular') {
                this.changeFrame(newFrame);
            } else if (prevButtonType === 'filtered') {
                this.searchAnnotations(newFrame, startFrame);
            } else {
                this.searchEmptyFrame(newFrame, startFrame);
            }
        }
    };

    private onNextFrame = async (): Promise<void> => {
        const { nextButtonType } = this.state;
        const {
            frameNumber, jobInstance, playing, onSwitchPlay, showDeletedFrames,
        } = this.props;
        const { stopFrame } = jobInstance;

        const frameFrom = Math.min(jobInstance.stopFrame, frameNumber + 1);
        const newFrame = await jobInstance.frames.search(
            { notDeleted: !showDeletedFrames },
            frameFrom,
            jobInstance.stopFrame,
        );
        if (newFrame !== frameNumber && newFrame !== null) {
            if (playing) {
                onSwitchPlay(false);
            }

            if (nextButtonType === 'regular') {
                this.changeFrame(newFrame);
            } else if (nextButtonType === 'filtered') {
                this.searchAnnotations(newFrame, stopFrame);
            } else {
                this.searchEmptyFrame(newFrame, stopFrame);
            }
        }
    };

    private onForward = async (): Promise<void> => {
        const {
            frameNumber, frameStep, jobInstance, playing, onSwitchPlay, showDeletedFrames,
        } = this.props;

        const newFrame = await jobInstance.frames.search(
            { notDeleted: !showDeletedFrames, offset: frameStep },
            Math.min(jobInstance.stopFrame, frameNumber + 1),
            jobInstance.stopFrame,
        );

        if (newFrame !== frameNumber && newFrame !== null) {
            if (playing) {
                onSwitchPlay(false);
            }
            this.changeFrame(newFrame);
        }
    };

    private onLastFrame = async (): Promise<void> => {
        const {
            frameNumber, jobInstance, playing, onSwitchPlay, showDeletedFrames,
        } = this.props;

        const newFrame =
            await jobInstance.frames.search({ notDeleted: !showDeletedFrames }, jobInstance.stopFrame, frameNumber);
        if (newFrame !== frameNumber && frameNumber !== null) {
            if (playing) {
                onSwitchPlay(false);
            }
            this.changeFrame(newFrame);
        }
    };

    private onSetPreviousButtonType = (type: 'regular' | 'filtered' | 'empty'): void => {
        this.setState({
            prevButtonType: type,
        });
    };

    private onSetNextButtonType = (type: 'regular' | 'filtered' | 'empty'): void => {
        this.setState({
            nextButtonType: type,
        });
    };

    private onSaveAnnotation = (): void => {
        const { onSaveAnnotation, jobInstance } = this.props;
        onSaveAnnotation(jobInstance);
    };

    private onChangePlayerSliderValue = async (value: number): Promise<void> => {
        const {
            playing, onSwitchPlay, jobInstance, showDeletedFrames,
        } = this.props;
        if (playing) {
            onSwitchPlay(false);
        }
        const newFrame = await jobInstance.frames.search(
            { notDeleted: !showDeletedFrames },
            Math.min(jobInstance.stopFrame, value),
            jobInstance.stopFrame,
        );
        if (newFrame !== null) {
            this.changeFrame(newFrame);
        }
    };

    private onChangePlayerInputValue = async (value: number): Promise<void> => {
        const {
            frameNumber, onSwitchPlay, playing, jobInstance, showDeletedFrames,
        } = this.props;

        if (value !== frameNumber) {
            if (playing) {
                onSwitchPlay(false);
            }
            const newFrame = await jobInstance.frames.search(
                { notDeleted: !showDeletedFrames },
                Math.min(jobInstance.stopFrame, value),
                jobInstance.stopFrame,
            );
            if (newFrame !== null) {
                this.changeFrame(newFrame);
            }
        }
    };

    private onFinishDraw = (): void => {
        const { activeControl, canvasInstance } = this.props;
        if (
            [ActiveControl.AI_TOOLS, ActiveControl.OPENCV_TOOLS].includes(activeControl) &&
            canvasInstance instanceof Canvas
        ) {
            canvasInstance.interact({ enabled: false });
            return;
        }

        canvasInstance.draw({ enabled: false });
    };

    private onSwitchToolsBlockerState = (): void => {
        const {
            toolsBlockerState, onSwitchToolsBlockerState, canvasInstance, activeControl,
        } = this.props;
        if (canvasInstance instanceof Canvas) {
            if (activeControl.includes(ActiveControl.OPENCV_TOOLS)) {
                canvasInstance.interact({
                    enabled: true,
                    crosshair: toolsBlockerState.algorithmsLocked,
                    enableThreshold: toolsBlockerState.algorithmsLocked,
                });
            }
        }
        onSwitchToolsBlockerState({ algorithmsLocked: !toolsBlockerState.algorithmsLocked });
    };

    private onURLIconClick = (): void => {
        const { frameNumber } = this.props;
        const { origin, pathname } = window.location;
        const url = `${origin}${pathname}?frame=${frameNumber}`;
        copy(url);
    };

    private onDeleteFrame = (): void => {
        const { deleteFrame, frameNumber, jobInstance } = this.props;
        if (jobInstance.type !== JobType.GROUND_TRUTH) deleteFrame(frameNumber);
    };

    private onRestoreFrame = (): void => {
        const { restoreFrame, frameNumber } = this.props;
        restoreFrame(frameNumber);
    };

    private beforeUnloadCallback = (event: BeforeUnloadEvent): string | undefined => {
        const { jobInstance, forceExit, setForceExitAnnotationFlag } = this.props;
        if (jobInstance.annotations.hasUnsavedChanges() && !forceExit) {
            const confirmationMessage = 'You have unsaved changes, please confirm leaving this page.';
            // eslint-disable-next-line no-param-reassign
            event.returnValue = confirmationMessage;
            return confirmationMessage;
        }

        if (forceExit) {
            setForceExitAnnotationFlag(false);
        }
        return undefined;
    };

    private play(): void {
        const {
            jobInstance,
            frameSpeed,
            frameNumber,
            frameDelay,
            frameFetching,
            playing,
            canvasIsReady,
            onSwitchPlay,
            onChangeFrame,
        } = this.props;

        if (playing && canvasIsReady && !frameFetching) {
            if (frameNumber < jobInstance.stopFrame) {
                let framesSkipped = 0;
                if (frameSpeed === FrameSpeed.Fast && frameNumber + 1 < jobInstance.stopFrame) {
                    framesSkipped = 1;
                }
                if (frameSpeed === FrameSpeed.Fastest && frameNumber + 2 < jobInstance.stopFrame) {
                    framesSkipped = 2;
                }

                setTimeout(async () => {
                    const { playing: stillPlaying } = this.props;
                    if (stillPlaying) {
                        if (isAbleToChangeFrame()) {
                            if (jobInstance.type === JobType.GROUND_TRUTH) {
                                const newFrame = await jobInstance.frames.search(
                                    { notDeleted: true },
                                    frameNumber + 1,
                                    jobInstance.stopFrame,
                                );
                                if (newFrame !== null) {
                                    onChangeFrame(newFrame, stillPlaying);
                                } else {
                                    onSwitchPlay(false);
                                }
                            } else {
                                onChangeFrame(frameNumber + 1 + framesSkipped, stillPlaying, framesSkipped + 1);
                            }
                        } else if (jobInstance.dimension === DimensionType.DIMENSION_2D) {
                            onSwitchPlay(false);
                        } else {
                            setTimeout(() => this.play(), frameDelay);
                        }
                    }
                }, frameDelay);
            } else {
                onSwitchPlay(false);
            }
        }
    }

    private autoSave(): void {
        const { autoSave, saving } = this.props;

        if (autoSave && !saving) {
            this.onSaveAnnotation();
        }
    }

    private changeFrame(frame: number): void {
        const { onChangeFrame } = this.props;
        if (isAbleToChangeFrame()) {
            onChangeFrame(frame);
        }
    }

    private searchAnnotations(start: number, stop: number): void {
        const { jobInstance, searchAnnotations } = this.props;
        if (isAbleToChangeFrame()) {
            searchAnnotations(jobInstance, start, stop);
        }
    }

    private searchEmptyFrame(start: number, stop: number): void {
        const { jobInstance, searchEmptyFrame } = this.props;
        if (isAbleToChangeFrame()) {
            searchEmptyFrame(jobInstance, start, stop);
        }
    }

    public render(): JSX.Element {
        const { nextButtonType, prevButtonType } = this.state;
        const {
            playing,
            saving,
            jobInstance,
            jobInstance: { startFrame, stopFrame },
            frameNumber,
            frameFilename,
            frameIsDeleted,
            undoAction,
            redoAction,
            workspace,
            canvasIsReady,
            keyMap,
            ranges,
            normalizedKeyMap,
            activeControl,
            searchAnnotations,
            changeWorkspace,
            switchNavigationBlocked,
            toolsBlockerState,
        } = this.props;

        const preventDefault = (event: KeyboardEvent | undefined): void => {
            if (event) {
                event.preventDefault();
            }
        };

        const subKeyMap = {
            SAVE_JOB: keyMap.SAVE_JOB,
            UNDO: keyMap.UNDO,
            REDO: keyMap.REDO,
            DELETE_FRAME: keyMap.DELETE_FRAME,
            NEXT_FRAME: keyMap.NEXT_FRAME,
            PREV_FRAME: keyMap.PREV_FRAME,
            FORWARD_FRAME: keyMap.FORWARD_FRAME,
            BACKWARD_FRAME: keyMap.BACKWARD_FRAME,
            SEARCH_FORWARD: keyMap.SEARCH_FORWARD,
            SEARCH_BACKWARD: keyMap.SEARCH_BACKWARD,
            PLAY_PAUSE: keyMap.PLAY_PAUSE,
            FOCUS_INPUT_FRAME: keyMap.FOCUS_INPUT_FRAME,
        };

        const handlers = {
            UNDO: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                if (undoAction) {
                    this.undo();
                }
            },
            REDO: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                if (redoAction) {
                    this.redo();
                }
            },
            SAVE_JOB: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                if (!saving) {
                    this.onSaveAnnotation();
                }
            },
            DELETE_FRAME: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                if (canvasIsReady) {
                    this.onDeleteFrame();
                }
            },
            NEXT_FRAME: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                if (canvasIsReady) {
                    this.onNextFrame();
                }
            },
            PREV_FRAME: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                if (canvasIsReady) {
                    this.onPrevFrame();
                }
            },
            FORWARD_FRAME: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                if (canvasIsReady) {
                    this.onForward();
                }
            },
            BACKWARD_FRAME: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                if (canvasIsReady) {
                    this.onBackward();
                }
            },
            SEARCH_FORWARD: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                if (frameNumber + 1 <= stopFrame && canvasIsReady && isAbleToChangeFrame()) {
                    searchAnnotations(jobInstance, frameNumber + 1, stopFrame);
                }
            },
            SEARCH_BACKWARD: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                if (frameNumber - 1 >= startFrame && canvasIsReady && isAbleToChangeFrame()) {
                    searchAnnotations(jobInstance, frameNumber - 1, startFrame);
                }
            },
            PLAY_PAUSE: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                this.onSwitchPlay();
            },
            FOCUS_INPUT_FRAME: (event: KeyboardEvent | undefined) => {
                preventDefault(event);
                if (this.inputFrameRef.current) {
                    this.inputFrameRef.current.focus();
                }
            },
        };

        return (
            <>
                <GlobalHotKeys keyMap={subKeyMap} handlers={handlers} />
                <AnnotationTopBarComponent
                    showStatistics={this.showStatistics}
                    showFilters={this.showFilters}
                    onSwitchPlay={this.onSwitchPlay}
                    onSaveAnnotation={this.onSaveAnnotation}
                    onPrevFrame={this.onPrevFrame}
                    onNextFrame={this.onNextFrame}
                    onForward={this.onForward}
                    onBackward={this.onBackward}
                    onFirstFrame={this.onFirstFrame}
                    onLastFrame={this.onLastFrame}
                    setNextButtonType={this.onSetNextButtonType}
                    setPrevButtonType={this.onSetPreviousButtonType}
                    onSliderChange={this.onChangePlayerSliderValue}
                    onInputChange={this.onChangePlayerInputValue}
                    onURLIconClick={this.onURLIconClick}
                    onDeleteFrame={this.onDeleteFrame}
                    onRestoreFrame={this.onRestoreFrame}
                    changeWorkspace={changeWorkspace}
                    switchNavigationBlocked={switchNavigationBlocked}
                    workspace={workspace}
                    playing={playing}
                    saving={saving}
                    ranges={ranges}
                    startFrame={startFrame}
                    stopFrame={stopFrame}
                    frameNumber={frameNumber}
                    frameFilename={frameFilename}
                    frameDeleted={frameIsDeleted}
                    inputFrameRef={this.inputFrameRef}
                    undoAction={undoAction}
                    redoAction={redoAction}
                    saveShortcut={normalizedKeyMap.SAVE_JOB}
                    undoShortcut={normalizedKeyMap.UNDO}
                    redoShortcut={normalizedKeyMap.REDO}
                    drawShortcut={normalizedKeyMap.SWITCH_DRAW_MODE}
                    // this shortcut is handled in interactionHandler.ts separately
                    switchToolsBlockerShortcut={normalizedKeyMap.SWITCH_TOOLS_BLOCKER_STATE}
                    playPauseShortcut={normalizedKeyMap.PLAY_PAUSE}
                    deleteFrameShortcut={normalizedKeyMap.DELETE_FRAME}
                    nextFrameShortcut={normalizedKeyMap.NEXT_FRAME}
                    previousFrameShortcut={normalizedKeyMap.PREV_FRAME}
                    forwardShortcut={normalizedKeyMap.FORWARD_FRAME}
                    backwardShortcut={normalizedKeyMap.BACKWARD_FRAME}
                    nextButtonType={nextButtonType}
                    prevButtonType={prevButtonType}
                    focusFrameInputShortcut={normalizedKeyMap.FOCUS_INPUT_FRAME}
                    onUndoClick={this.undo}
                    onRedoClick={this.redo}
                    onFinishDraw={this.onFinishDraw}
                    onSwitchToolsBlockerState={this.onSwitchToolsBlockerState}
                    toolsBlockerState={toolsBlockerState}
                    jobInstance={jobInstance}
                    activeControl={activeControl}
                    deleteFrameAvailable={jobInstance.type !== JobType.GROUND_TRUTH}
                />
            </>
        );
    }
}

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(AnnotationTopBarContainer));
