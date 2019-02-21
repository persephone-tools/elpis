import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import { Grid, Header, Segment, Form, Input, Button } from 'semantic-ui-react';
import { connect } from 'react-redux';
import { translate } from 'react-i18next';
import { Formik } from 'formik';
import { modelName } from 'redux/actions';
import Branding from 'components/Steps/Shared/Branding';
import Informer from 'components/Steps/Shared/Informer';
import urls from 'urls'

class ModelNew extends Component {

    render() {
        const { t, name, modelName } = this.props;
        return (
            <div>
                <Branding />
                <Segment>
                    <Grid centered>
                        <Grid.Column width={ 4 }>
                            <Informer />
                        </Grid.Column>

                        <Grid.Column width={ 12 }>
                            <Header as='h1' text="true">
                                { t('model.new.title') }
                            </Header>

                            <Formik
                                enableReinitialize
                                initialValues={ {
                                    name: name
                                } }
                                validate={ values => {
                                    let errors = {};
                                    if (!values.name) {
                                        errors.name = 'Required';
                                    } else if (
                                        !/^[A-Za-z ]+$/i.test(values.name)
                                    ) {
                                        errors.name = 'Invalid name';
                                    }
                                    return errors;
                                } }
                                onSubmit={ (values, { setSubmitting }) => {
                                    // demo
                                    setTimeout(() => {
                                        alert(JSON.stringify(values, null, 2));
                                        setSubmitting(false);
                                    }, 400);

                                    // redux action
                                    modelName(values.name)

                                    // go to next page
                                    this.props.history.push(urls.gui.model.pronunciation)
                                } }
                            >
                                { ({
                                    values,
                                    errors,
                                    dirty,
                                    touched,
                                    handleSubmit,
                                    handleChange,
                                    isSubmitting,
                                    /* and other goodies */
                                }) => (
                                        <Form onSubmit={ handleSubmit }>
                                            <Form.Field>
                                                <Input
                                                    label={ t('model.new.nameLabel') }
                                                    value={ values.name }
                                                    placeholder={ t('model.new.namePlaceholder') }
                                                    name="name"
                                                    type="text"
                                                    onChange={ handleChange } />
                                            </Form.Field>
                                            <Button type='submit' onClick={ handleSubmit } >
                                                { t('model.new.nextButton') }
                                            </Button>
                                        </Form>
                                    ) }
                            </Formik>
                        </Grid.Column>
                    </Grid>
                </Segment>
            </div>
        )
    }
}

const mapStateToProps = state => {
    return {
        name: state.model.name
    }
}
const mapDispatchToProps = dispatch => ({
    modelName: name => {
        dispatch(modelName(name))
    }
})
export default connect(mapStateToProps, mapDispatchToProps)(translate('common')(ModelNew));
